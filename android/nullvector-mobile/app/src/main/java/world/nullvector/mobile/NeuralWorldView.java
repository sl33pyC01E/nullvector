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
import android.view.MotionEvent;

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
    private volatile double actionMilliseconds = 0;
    private volatile Bitmap neuralFrame;
    private volatile boolean running = true;
    private volatile float controlX = 0f, controlY = 0f;
    private volatile int actionId = 0;
    private volatile boolean movementTouch = false, actionTouch = false;

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

    private float[] floatAsset(String name, int count) throws Exception {
        byte[] bytes; try (InputStream input = getContext().getAssets().open(name)) { bytes = input.readAllBytes(); }
        FloatBuffer floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        if (floats.remaining() != count) throw new IllegalStateException(name + " length"); float[] value = new float[count]; floats.get(value); return value;
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
            String actionModel = BuildConfig.SPLIT_ACTION ? "action_delta_int8_qdq.onnx" : "action_core_fp32.onnx";
            try (OrtSession actionSession = environment.createSession(assetFile(actionModel).getAbsolutePath(), options);
                 OrtSession actorSession = BuildConfig.SPLIT_ACTION ? environment.createSession(assetFile("actor_state_fp32.onnx").getAbsolutePath(), options) : null;
                 OrtSession decoder = environment.createSession(assetFile("frame_vae_fp32.onnx").getAbsolutePath(), options)) {
                float[] rawInitial = latentAsset(), latentNorm = floatAsset("latent_normalization.f32", 96), current = new float[rawInitial.length];
                for (int c = 0, i = 0; c < 48; c++) for (int p = 0; p < 32 * 32; p++, i++) current[i] = (rawInitial[i] - latentNorm[c]) / latentNorm[48 + c];
                float[] previous = current.clone(), actor = new float[128], previousActor = new float[128];
                float[] control = new float[4], visibility = new float[32 * 32], memory = new float[32 * 32];
                java.util.Arrays.fill(visibility, 1f); int tick = 0;
                status += BuildConfig.SPLIT_ACTION
                    ? " · INT8 action + FP32 physiology + mobile VAE live"
                    : " · FP32 action + physiology + mobile VAE live";
                while (running) {
                    long frameBegan = System.nanoTime(); control[0] = controlX; control[1] = controlY; control[2] = actionTouch ? 1f : 0f; control[3] = 0f;
                    Map<String, OnnxTensor> actionInputs = new HashMap<>();
                    actionInputs.put("current", OnnxTensor.createTensor(environment, FloatBuffer.wrap(current), new long[]{1, 48, 32, 32}));
                    actionInputs.put("previous", OnnxTensor.createTensor(environment, FloatBuffer.wrap(previous), new long[]{1, 48, 32, 32}));
                    actionInputs.put("action", OnnxTensor.createTensor(environment, LongBuffer.wrap(new long[]{actionId}), new long[]{1}));
                    actionInputs.put("control", OnnxTensor.createTensor(environment, FloatBuffer.wrap(control), new long[]{1, 4}));
                    actionInputs.put("context", OnnxTensor.createTensor(environment, FloatBuffer.wrap(context), new long[]{1, 64}));
                    actionInputs.put("actor", OnnxTensor.createTensor(environment, FloatBuffer.wrap(actor), new long[]{1, 128}));
                    if (!BuildConfig.SPLIT_ACTION) actionInputs.put("previous_actor", OnnxTensor.createTensor(environment, FloatBuffer.wrap(previousActor), new long[]{1, 128}));
                    actionInputs.put("visibility", OnnxTensor.createTensor(environment, FloatBuffer.wrap(visibility), new long[]{1, 1, 32, 32}));
                    actionInputs.put("memory", OnnxTensor.createTensor(environment, FloatBuffer.wrap(memory), new long[]{1, 1, 32, 32}));
                    float[] next = current.clone();
                    float[] proposedActor = null, actorGate = null;
                    long actionBegan = System.nanoTime();
                    try (OrtSession.Result result = actionSession.run(actionInputs)) {
                        float[][][][] delta = (float[][][][])result.get(0).getValue(); float[][][][] gate = (float[][][][])result.get(1).getValue();
                        float bias = 1.5f * Math.min(tick / 2f, 1f);
                        previous = current; for (int c = 0, i = 0; c < 48; c++) for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++, i++) next[i] += (float)(1.0 / (1.0 + Math.exp(-(gate[0][gate[0].length == 1 ? 0 : c][y][x] + bias)))) * delta[0][c][y][x];
                        if (!BuildConfig.SPLIT_ACTION) {
                            proposedActor = ((float[][])result.get(2).getValue())[0].clone();
                            actorGate = ((float[][])result.get(3).getValue())[0].clone();
                        }
                    }
                    actionMilliseconds = (System.nanoTime() - actionBegan) / 1_000_000.0; current = next;
                    for (OnnxTensor tensor : actionInputs.values()) tensor.close();
                    if (BuildConfig.SPLIT_ACTION) {
                        Map<String, OnnxTensor> actorInputs = new HashMap<>();
                        actorInputs.put("actor", OnnxTensor.createTensor(environment, FloatBuffer.wrap(actor), new long[]{1, 128})); actorInputs.put("previous_actor", OnnxTensor.createTensor(environment, FloatBuffer.wrap(previousActor), new long[]{1, 128})); actorInputs.put("action", OnnxTensor.createTensor(environment, LongBuffer.wrap(new long[]{actionId}), new long[]{1})); actorInputs.put("control", OnnxTensor.createTensor(environment, FloatBuffer.wrap(control), new long[]{1, 4})); actorInputs.put("context", OnnxTensor.createTensor(environment, FloatBuffer.wrap(context), new long[]{1, 64})); actorInputs.put("visibility", OnnxTensor.createTensor(environment, FloatBuffer.wrap(visibility), new long[]{1, 1, 32, 32})); actorInputs.put("memory", OnnxTensor.createTensor(environment, FloatBuffer.wrap(memory), new long[]{1, 1, 32, 32}));
                        try (OrtSession.Result result = actorSession.run(actorInputs)) { proposedActor = ((float[][])result.get(0).getValue())[0].clone(); actorGate = ((float[][])result.get(1).getValue())[0].clone(); }
                        for (OnnxTensor tensor : actorInputs.values()) tensor.close();
                    }
                    float[] nextActor = actor.clone();
                    for (int i = 0; i < actor.length; i++) if (actorGate[i] >= .7f) nextActor[i] += .9f * (proposedActor[i] - actor[i]);
                    previousActor = actor; actor = nextActor;
                    float[] rawLatent = new float[current.length]; for (int c = 0, i = 0; c < 48; c++) for (int p = 0; p < 32 * 32; p++, i++) rawLatent[i] = current[i] * latentNorm[48 + c] + latentNorm[c];
                    Map<String, OnnxTensor> decoderInput = new HashMap<>(); decoderInput.put("latent", OnnxTensor.createTensor(environment, FloatBuffer.wrap(rawLatent), new long[]{1, 48, 32, 32}));
                    long decoderBegan = System.nanoTime(); try (OrtSession.Result result = decoder.run(decoderInput)) { neuralFrame = bitmap((float[][][][])result.get(0).getValue()); }
                    decoderMilliseconds = (System.nanoTime() - decoderBegan) / 1_000_000.0; decoderInput.get("latent").close(); postInvalidateOnAnimation(); tick++;
                    long remaining = 33_333_333L - (System.nanoTime() - frameBegan); if (remaining > 0) Thread.sleep(remaining / 1_000_000L, (int)(remaining % 1_000_000L));
                }
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
        paint.setTextSize(20); paint.setColor(Color.rgb(170, 195, 202)); canvas.drawText(status, 42, 88, paint); canvas.drawText(String.format("context %.2f ms · action %.2f ms · raster %.2f ms · target 30 FPS", milliseconds, actionMilliseconds, decoderMilliseconds), 42, 118, paint);
        paint.setTextSize(16); canvas.drawText("left touch: move · right touch: action selection/act · recurrent context → action → VAE", 42, height - 42, paint);
        paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(3); paint.setColor(movementTouch ? Color.rgb(67, 239, 220) : Color.rgb(55, 85, 92)); canvas.drawCircle(width * .16f, height * .82f, 72, paint);
        paint.setStyle(Paint.Style.FILL); canvas.drawCircle(width * .16f + controlX * 58, height * .82f + controlY * 58, 14, paint);
        paint.setColor(actionTouch ? Color.rgb(255, 80, 180) : Color.rgb(80, 55, 75)); canvas.drawCircle(width * .84f, height * .82f, 62, paint); paint.setColor(Color.WHITE); paint.setTextSize(18); canvas.drawText("A" + actionId, width * .84f - 15, height * .82f + 6, paint);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        boolean move = false, act = false; float width = Math.max(1, getWidth()), height = Math.max(1, getHeight());
        if (event.getActionMasked() != MotionEvent.ACTION_UP && event.getActionMasked() != MotionEvent.ACTION_CANCEL) for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            float x = event.getX(pointer), y = event.getY(pointer);
            if (x < width * .55f) { controlX = Math.max(-1, Math.min(1, (x - width * .16f) / 72f)); controlY = Math.max(-1, Math.min(1, (y - height * .82f) / 72f)); move = true; }
            else { actionId = Math.max(0, Math.min(21, (int)(y / height * 22))); act = true; }
        }
        movementTouch = move; actionTouch = act; if (!move) { controlX = 0; controlY = 0; } invalidate(); return true;
    }

    @Override protected void onDetachedFromWindow() { running = false; super.onDetachedFromWindow(); }
}
