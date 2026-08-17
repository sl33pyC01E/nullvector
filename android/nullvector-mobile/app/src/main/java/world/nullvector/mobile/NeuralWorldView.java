package world.nullvector.mobile;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

public final class NeuralWorldView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private volatile String status = "Loading neural world context…";
    private volatile float[] context = new float[64];
    private volatile double milliseconds = 0;

    public NeuralWorldView(Context owner) {
        super(owner); paint.setTypeface(android.graphics.Typeface.MONOSPACE);
        new Thread(this::runContextModel, "nullvector-context").start();
    }

    private File assetFile(String name) throws Exception {
        File target = new File(getContext().getFilesDir(), name);
        if (!target.isFile()) try (InputStream input = getContext().getAssets().open(name); FileOutputStream output = new FileOutputStream(target)) {
            input.transferTo(output);
        }
        return target;
    }

    private void runContextModel() {
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
        } catch (Throwable failure) { status = "Model load failed: " + failure.getClass().getSimpleName(); }
        postInvalidate();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); canvas.drawColor(Color.rgb(4, 8, 13)); float width = getWidth(), height = getHeight();
        paint.setColor(Color.rgb(24, 45, 54)); paint.setStrokeWidth(1);
        for (int x = 0; x < width; x += 32) canvas.drawLine(x, 0, x, height, paint);
        for (int y = 0; y < height; y += 32) canvas.drawLine(0, y, width, y, paint);
        float cx = width * .5f, cy = height * .52f, radius = Math.min(width, height) * .31f;
        for (int i = 0; i < context.length; i++) {
            double angle = i * Math.PI * 2 / context.length; float strength = Math.min(1, Math.abs(context[i]));
            float x = cx + (float)Math.cos(angle) * radius * (.55f + .4f * strength); float y = cy + (float)Math.sin(angle) * radius * (.55f + .4f * strength);
            paint.setColor(Color.rgb(20 + (int)(40 * strength), 130 + (int)(110 * strength), 145 + (int)(90 * strength)));
            canvas.drawCircle(x, y, 3 + 10 * strength, paint);
        }
        paint.setTextSize(28); paint.setColor(Color.rgb(67, 239, 220)); canvas.drawText("NULLVECTOR // GALAXY S25 ULTRA", 42, 54, paint);
        paint.setTextSize(20); paint.setColor(Color.rgb(170, 195, 202)); canvas.drawText(status, 42, 88, paint); canvas.drawText(String.format("context %.2f ms · target 15 Hz", milliseconds), 42, 118, paint);
        paint.setTextSize(16); canvas.drawText("Android foundation: live neural context · action core and VAE staged for device profiling", 42, height - 42, paint);
    }
}
