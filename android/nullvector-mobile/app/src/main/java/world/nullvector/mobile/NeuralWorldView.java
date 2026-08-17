package world.nullvector.mobile;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Bitmap;
import android.graphics.Paint;
import android.view.View;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

public final class NeuralWorldView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private volatile String status = "Loading neural world context…";
    private volatile float[] context = new float[64];
    private volatile double milliseconds = 0;
    private volatile double decoderMilliseconds = 0;
    private volatile Bitmap neuralFrame;

    public NeuralWorldView(Context owner) {
        super(owner); paint.setTypeface(android.graphics.Typeface.MONOSPACE);
        new Thread(this::runModels, "nullvector-neural-runtime").start();
    }

    private File assetFile(String name) throws Exception {
        File target = new File(getContext().getFilesDir(), name);
        if (!target.isFile()) try (InputStream input = getContext().getAssets().open(name); FileOutputStream output = new FileOutputStream(target)) {
            input.transferTo(output);
        }
        return target;
    }

    private float[] latentAsset() throws Exception {
        byte[] bytes;
        try (InputStream input = getContext().getAssets().open("sample_latent.f32")) { bytes = input.readAllBytes(); }
        FloatBuffer floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] value = new float[48 * 32 * 32]; floats.get(value); return value;
    }

    private Bitmap bitmap(float[][][][] rgb) {
        int[] pixels = new int[256 * 256];
        for (int y = 0; y < 256; y++) for (int x = 0; x < 256; x++) {
            int r = Math.max(0, Math.min(255, Math.round(rgb[0][0][y][x] * 255)));
            int g = Math.max(0, Math.min(255, Math.round(rgb[0][1][y][x] * 255)));
            int b = Math.max(0, Math.min(255, Math.round(rgb[0][2][y][x] * 255)));
            pixels[y * 256 + x] = Color.rgb(r, g, b);
        }
        return Bitmap.createBitmap(pixels, 256, 256, Bitmap.Config.ARGB_8888);
    }

    private void runModels() {
        try (OrtEnvironment environment = OrtEnvironment.getEnvironment(); OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            String provider = "ORT CPU";
            try { options.addNnapi(); provider = "NNAPI"; } catch (Throwable ignored) { }
            try (OrtSession session = environment.createSession(assetFile("world_context_fp32.onnx").getAbsolutePath(), options)) {
                long[] categorical = new long[32 * 32]; float[] continuous = new float[7 * 32 * 32]; float[] condition = new float[15]; condition[0] = condition[6] = 1f;
                for (int i = 0; i < continuous.length; i++) continuous[i] = .25f + .25f * (float)Math.sin(i * .019);
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("terrain", OnnxTensor.createTensor(environment, LongBuffer.wrap(categorical), new long[]{1, 32, 32}));
                inputs.put("city", OnnxTensor.createTensor(environment, LongBuffer.wrap(categorical), new long[]{1, 32, 32}));
                inputs.put("continuous", OnnxTensor.createTensor(environment, FloatBuffer.wrap(continuous), new long[]{1, 7, 32, 32}));
                inputs.put("condition", OnnxTensor.createTensor(environment, FloatBuffer.wrap(condition), new long[]{1, 15}));
                for (int warmup = 0; warmup < 4; warmup++) try (OrtSession.Result ignored = session.run(inputs)) { }
                long began = System.nanoTime();
                try (OrtSession.Result result = session.run(inputs)) { context = ((float[][])result.get(0).getValue())[0]; }
                milliseconds = (System.nanoTime() - began) / 1_000_000.0;
                for (OnnxTensor tensor : inputs.values()) tensor.close();
                status = provider + " · structured world encoder live";
            }
            try (OrtSession decoder = environment.createSession(assetFile("frame_vae_fp32.onnx").getAbsolutePath(), options)) {
                Map<String, OnnxTensor> input = new HashMap<>(); input.put("latent", OnnxTensor.createTensor(environment, FloatBuffer.wrap(latentAsset()), new long[]{1, 48, 32, 32}));
                for (int warmup = 0; warmup < 3; warmup++) try (OrtSession.Result ignored = decoder.run(input)) { }
                long began = System.nanoTime();
                try (OrtSession.Result result = decoder.run(input)) { neuralFrame = bitmap((float[][][][])result.get(0).getValue()); }
                decoderMilliseconds = (System.nanoTime() - began) / 1_000_000.0; input.get("latent").close(); status += " · mobile VAE live";
            }
        } catch (Throwable failure) { status = "Model load failed: " + failure.getClass().getSimpleName(); }
        postInvalidate();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); canvas.drawColor(Color.rgb(4, 8, 13)); float width = getWidth(), height = getHeight();
        paint.setColor(Color.rgb(24, 45, 54)); paint.setStrokeWidth(1);
        for (int x = 0; x < width; x += 32) canvas.drawLine(x, 0, x, height, paint);
        for (int y = 0; y < height; y += 32) canvas.drawLine(0, y, width, y, paint);
        float cx = width * .25f, cy = height * .52f, radius = Math.min(width * .5f, height) * .31f;
        for (int i = 0; i < context.length; i++) {
            double angle = i * Math.PI * 2 / context.length; float strength = Math.min(1, Math.abs(context[i]));
            float x = cx + (float)Math.cos(angle) * radius * (.55f + .4f * strength); float y = cy + (float)Math.sin(angle) * radius * (.55f + .4f * strength);
            paint.setColor(Color.rgb(20 + (int)(40 * strength), 130 + (int)(110 * strength), 145 + (int)(90 * strength)));
            canvas.drawCircle(x, y, 3 + 10 * strength, paint);
        }
        if (neuralFrame != null) {
            float size = Math.min(height * .72f, width * .43f); paint.setFilterBitmap(true);
            canvas.drawBitmap(neuralFrame, null, new android.graphics.RectF(width * .73f - size * .5f, cy - size * .5f, width * .73f + size * .5f, cy + size * .5f), paint);
            paint.setFilterBitmap(false);
        }
        paint.setTextSize(28); paint.setColor(Color.rgb(67, 239, 220)); canvas.drawText("NULLVECTOR // GALAXY S25 ULTRA", 42, 54, paint);
        paint.setTextSize(20); paint.setColor(Color.rgb(170, 195, 202)); canvas.drawText(status, 42, 88, paint); canvas.drawText(String.format("context %.2f ms · raster %.2f ms · target 30 FPS", milliseconds, decoderMilliseconds), 42, 118, paint);
        paint.setTextSize(16); canvas.drawText("Android foundation: live structured context + distilled neural raster · action core staged for device profiling", 42, height - 42, paint);
    }
}
