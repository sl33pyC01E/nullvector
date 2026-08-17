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
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;

public final class NeuralWorldView extends View {
    private static final String TAG = "NullvectorRuntime";
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private volatile String status = "Loading neural world context…";
    private volatile float[] context = new float[64];
    private volatile double milliseconds = 0;
    private volatile double decoderMilliseconds = 0;
    private volatile double actionMilliseconds = 0;
    private volatile double cellularMilliseconds = 0;
    private volatile Bitmap neuralFrame;
    private volatile Bitmap cellularFrame;
    private volatile float cellularHealth = 1f;
    private volatile float cellularNeural = 1f;
    private volatile boolean running = true;
    private volatile float controlX = 0f, controlY = 0f;
    private volatile float aimX = 1f, aimY = 0f;
    private volatile float worldX = 2048f, worldY = 2048f;
    private volatile float velocityX = 0f, velocityY = 0f;
    private volatile boolean diagnostics = false;
    private volatile int gathered = 0;
    private volatile int fireRequests = 0;
    private volatile float pendingNutrition = 0f;
    private volatile int actionId = 0;
    private volatile boolean movementTouch = false, actionTouch = false;
    private boolean actionWasDown = false;
    private final List<Projectile> projectiles = new ArrayList<>();
    private final List<MaterialNode> materials = new ArrayList<>();

    private static final class Projectile {
        float x, y, z, vx, vy, vz; int bounces; boolean resting;
        Projectile(float x, float y, float aimX, float aimY) {
            this.x = x; this.y = y; this.z = 54f; this.vx = aimX * 680f; this.vy = aimY * 680f; this.vz = 255f;
        }
    }

    private static final class MaterialNode {
        float x, y, amount; final int type;
        MaterialNode(float x, float y, int type, float amount) { this.x = x; this.y = y; this.type = type; this.amount = amount; }
    }

    public NeuralWorldView(Context owner) {
        super(owner); paint.setTypeface(android.graphics.Typeface.MONOSPACE);
        for (int i = 0; i < 180; i++) { long hash = worldHash(i, i * 37 + 11); materials.add(new MaterialNode(90 + Math.floorMod(hash, 3916), 90 + Math.floorMod(hash >>> 21, 3916), (int)Math.floorMod(hash >>> 42, 3), .55f + Math.floorMod(hash >>> 49, 45) / 100f)); }
        materials.add(new MaterialNode(2165, 1985, 0, 1f)); materials.add(new MaterialNode(1905, 2115, 1, 1f)); materials.add(new MaterialNode(2250, 2180, 2, 1f));
        new Thread(this::runModels, "nullvector-neural-runtime").start();
    }

    private File assetFile(String name) throws Exception {
        File root = new File(getContext().getFilesDir(), "models-v" + BuildConfig.VERSION_CODE + (BuildConfig.SPLIT_ACTION ? "-int8" : "-fp32"));
        if (!root.isDirectory() && !root.mkdirs()) throw new IllegalStateException("model cache directory");
        File target = new File(root, name);
        if (!target.isFile()) {
            File temporary = new File(root, name + ".partial");
            if (temporary.exists() && !temporary.delete()) throw new IllegalStateException("stale model partial");
            try (InputStream input = getContext().getAssets().open(name); FileOutputStream output = new FileOutputStream(temporary)) {
                input.transferTo(output); output.getFD().sync();
            }
            if (temporary.length() == 0 || !temporary.renameTo(target)) throw new IllegalStateException("model publish " + name);
        }
        return target;
    }

    private void stage(String value) {
        status = value; Log.i(TAG, value); postInvalidate();
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

    private Bitmap cellularBitmap(float[] anatomy, float[] state) {
        int cells = 48 * 48;
        int[] pixels = new int[cells];
        for (int p = 0; p < cells; p++) {
            float body = anatomy[p], health = state[p], fluid = state[cells + p];
            float energy = state[3 * cells + p], oxygen = state[4 * cells + p];
            float scar = state[6 * cells + p], wound = state[7 * cells + p];
            float neural = state[8 * cells + p], surface = state[9 * cells + p];
            float biomass = state[10 * cells + p], alive = state[11 * cells + p];
            float circulation = Math.min(1f, anatomy[(28 * cells) + p] + anatomy[(29 * cells) + p] + anatomy[(30 * cells) + p]);
            float respiration = Math.min(1f, anatomy[(31 * cells) + p] + anatomy[(32 * cells) + p] + anatomy[(33 * cells) + p]);
            float digestion = Math.min(1f, anatomy[(34 * cells) + p] + anatomy[(35 * cells) + p] + anatomy[(36 * cells) + p]);
            float neuralOrgan = Math.min(1f, anatomy[(37 * cells) + p] + anatomy[(38 * cells) + p] + anatomy[(39 * cells) + p]);
            int r = Math.round(body * alive * (25 + 105 * health + 100 * wound + 55 * scar + 60 * biomass + 90 * circulation));
            int g = Math.round(body * alive * (42 + 120 * health + 75 * energy + 80 * digestion + 75 * respiration) + 100 * surface);
            int b = Math.round(body * alive * (55 + 125 * health + 75 * fluid + 70 * oxygen + 115 * neural + 100 * neuralOrgan) + 220 * surface);
            int alpha = Math.round(Math.max(body * alive, Math.min(1f, surface * 2f)) * 255);
            pixels[p] = Color.argb(Math.max(0, Math.min(255, alpha)), Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b)));
        }
        return Bitmap.createBitmap(pixels, 48, 48, Bitmap.Config.ARGB_8888);
    }

    private float bodyMean(float[] anatomy, float[] state, int channel) {
        int cells = 48 * 48, offset = channel * cells; float total = 0f, count = 0f;
        for (int p = 0; p < cells; p++) if (anatomy[p] > .5f) { total += state[offset + p]; count += 1f; }
        return count > 0 ? total / count : 0f;
    }

    private float organMean(float[] anatomy, float[] state, int channel, int staticStart) {
        int cells = 48 * 48, offset = channel * cells; float total = 0f, count = 0f;
        for (int p = 0; p < cells; p++) {
            float mask = Math.min(1f, anatomy[staticStart * cells + p] + anatomy[(staticStart + 1) * cells + p] + anatomy[(staticStart + 2) * cells + p]);
            total += state[offset + p] * mask; count += mask;
        }
        return count > 0 ? total / count : 0f;
    }

    private static long worldHash(int x, int y) {
        long value = (x * 0x9E3779B97F4A7C15L) ^ (y * 0xC2B2AE3D27D4EB4FL) ^ 0x4E554C4C56454354L;
        value ^= value >>> 30; value *= 0xBF58476D1CE4E5B9L; value ^= value >>> 27; value *= 0x94D049BB133111EBL; return value ^ (value >>> 31);
    }

    private void advanceHabitat(float dt) {
        float targetX = controlX * 250f, targetY = controlY * 205f;
        velocityX += (targetX - velocityX) * Math.min(1f, dt * 9f); velocityY += (targetY - velocityY) * Math.min(1f, dt * 9f);
        worldX = Math.max(80f, Math.min(4016f, worldX + velocityX * dt)); worldY = Math.max(80f, Math.min(4016f, worldY + velocityY * dt));
        if (fireRequests > 0 || (actionTouch && !actionWasDown)) synchronized (projectiles) { projectiles.add(new Projectile(worldX, worldY - 18f, aimX, aimY)); if (fireRequests > 0) fireRequests--; }
        actionWasDown = actionTouch;
        synchronized (projectiles) {
            for (Projectile shot : projectiles) if (!shot.resting) {
                shot.x += shot.vx * dt; shot.y += shot.vy * dt; shot.z += shot.vz * dt; shot.vz -= 560f * dt;
                if (shot.z <= 0f) {
                    shot.z = 0f;
                    if (shot.bounces < 2 && Math.abs(shot.vz) > 80f) { shot.vz = -shot.vz * .34f; shot.vx *= .72f; shot.vy *= .72f; shot.bounces++; }
                    else { shot.vz = 0f; shot.vx *= .88f; shot.vy *= .88f; if (Math.hypot(shot.vx, shot.vy) < 18f) shot.resting = true; }
                }
                if (shot.z < 34f) synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f && Math.hypot(shot.x - node.x, shot.y - node.y) < 25f) { node.amount = Math.max(0f, node.amount - .42f); gathered++; shot.vx *= -.18f; shot.vy *= -.18f; shot.vz = Math.max(80f, shot.vz); break; } }
            }
            if (projectiles.size() > 48) projectiles.subList(0, projectiles.size() - 48).clear();
        }
        synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f && node.type != 2 && Math.hypot(worldX - node.x, worldY - node.y) < 27f) { float eaten = Math.min(node.amount, dt * .32f); node.amount -= eaten; pendingNutrition += eaten * (node.type == 0 ? 1f : .45f); gathered += node.amount <= 0f ? 1 : 0; } }
    }

    private void runModels() {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            // The first preview enabled NNAPI generically. On current Samsung
            // firmware that can take the whole process down inside the vendor
            // driver before Java receives an exception. Keep this recovery
            // build on ORT CPU until a model-by-model QNN partition is tested.
            options.setIntraOpNumThreads(2); options.setInterOpNumThreads(1);
            String provider = "ORT CPU SAFE";
            stage("Extracting neural world context…");
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
                stage(provider + " · structured world encoder live");
            }
            String actionModel = BuildConfig.SPLIT_ACTION ? "action_delta_int8_qdq.onnx" : "action_core_fp32.onnx";
            stage(provider + " · loading " + (BuildConfig.SPLIT_ACTION ? "INT8" : "FP32") + " action runtime…");
            try (OrtSession actionSession = environment.createSession(assetFile(actionModel).getAbsolutePath(), options);
                 OrtSession actorSession = BuildConfig.SPLIT_ACTION ? environment.createSession(assetFile("actor_state_fp32.onnx").getAbsolutePath(), options) : null;
                 OrtSession decoder = environment.createSession(assetFile("frame_vae_fp32.onnx").getAbsolutePath(), options);
                 OrtSession cellularSession = environment.createSession(assetFile("mobile_cell_nca_fp32.onnx").getAbsolutePath(), options)) {
                float[] rawInitial = latentAsset(), latentNorm = floatAsset("latent_normalization.f32", 96), current = new float[rawInitial.length];
                for (int c = 0, i = 0; c < 48; c++) for (int p = 0; p < 32 * 32; p++, i++) current[i] = (rawInitial[i] - latentNorm[c]) / latentNorm[48 + c];
                float[] previous = current.clone(), actor = new float[128], previousActor = new float[128];
                float[] control = new float[4], visibility = new float[32 * 32], memory = new float[32 * 32];
                float[] cellStatic = floatAsset("cell_static.f32", 85 * 48 * 48);
                float[] cellState = floatAsset("cell_state.f32", 12 * 48 * 48);
                float[] cellBonds = floatAsset("cell_bonds.f32", 8 * 48 * 48);
                java.util.Arrays.fill(visibility, 1f); int tick = 0;
                stage(provider + (BuildConfig.SPLIT_ACTION
                    ? " · INT8 action + cellular NCA + mobile VAE live"
                    : " · FP32 action + cellular NCA + mobile VAE live"));
                try (OnnxTensor cellStaticTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(cellStatic), new long[]{1, 85, 48, 48});
                     OnnxTensor cellBondTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(cellBonds), new long[]{1, 8, 48, 48})) {
                  while (running) {
                    long frameBegan = System.nanoTime(); advanceHabitat(1f / 30f); control[0] = controlX; control[1] = controlY; control[2] = actionTouch ? 1f : 0f; control[3] = Math.max(-1f, Math.min(1f, cellularHealth * cellularNeural * 2f - 1f));
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
                    if ((tick & 1) == 0) {
                        long cellularBegan = System.nanoTime();
                        float absorbed = pendingNutrition; pendingNutrition = 0f;
                        if (absorbed > 0f) for (int p = 0, cells = 48 * 48; p < cells; p++) if (cellStatic[p] > .5f) { cellState[2 * cells + p] = Math.min(1f, cellState[2 * cells + p] + absorbed * .018f); cellState[3 * cells + p] = Math.min(1f, cellState[3 * cells + p] + absorbed * .009f); }
                        try (OnnxTensor cellStateTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(cellState), new long[]{1, 12, 48, 48})) {
                            Map<String, OnnxTensor> cellInputs = new HashMap<>(); cellInputs.put("static", cellStaticTensor); cellInputs.put("state", cellStateTensor); cellInputs.put("live_bonds", cellBondTensor);
                            try (OrtSession.Result result = cellularSession.run(cellInputs)) {
                                float[][][][] value = (float[][][][])result.get(0).getValue(); int cursor = 0;
                                for (int c = 0; c < 12; c++) for (int y = 0; y < 48; y++) for (int x = 0; x < 48; x++) cellState[cursor++] = value[0][c][y][x];
                            }
                        }
                        cellularMilliseconds = (System.nanoTime() - cellularBegan) / 1_000_000.0;
                        cellularHealth = bodyMean(cellStatic, cellState, 0); cellularNeural = organMean(cellStatic, cellState, 8, 37); cellularFrame = cellularBitmap(cellStatic, cellState);
                    }
                    long remaining = 33_333_333L - (System.nanoTime() - frameBegan); if (remaining > 0) Thread.sleep(remaining / 1_000_000L, (int)(remaining % 1_000_000L));
                  }
                }
            }
        } catch (Throwable failure) {
            Log.e(TAG, "Neural runtime failed", failure);
            status = "SAFE FAILURE · " + failure.getClass().getSimpleName() + " · " + String.valueOf(failure.getMessage());
        }
        postInvalidate();
    }

    private void bar(Canvas canvas, float x, float y, float width, float value, int color, String label) {
        paint.setColor(Color.argb(180, 5, 11, 16)); canvas.drawRect(x, y, x + width, y + 18, paint); paint.setColor(color); canvas.drawRect(x + 2, y + 2, x + 2 + (width - 4) * Math.max(0f, Math.min(1f, value)), y + 16, paint); paint.setTextSize(13); paint.setColor(Color.WHITE); canvas.drawText(label, x + 6, y + 14, paint);
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); canvas.drawColor(Color.rgb(4, 9, 12)); float width = getWidth(), height = getHeight(); float cx = width * .5f, cy = height * .54f;
        int tile = 64, minX = (int)Math.floor((worldX - cx) / tile) - 1, maxX = (int)Math.ceil((worldX + cx) / tile) + 1;
        int minY = (int)Math.floor((worldY - cy) / tile) - 1, maxY = (int)Math.ceil((worldY + (height - cy)) / tile) + 1;
        for (int ty = minY; ty <= maxY; ty++) for (int tx = minX; tx <= maxX; tx++) {
            long hash = worldHash(tx, ty); int kind = (int)Math.floorMod(hash, 17); float sx = cx + tx * tile - worldX, sy = cy + ty * tile - worldY;
            int base = 13 + (int)Math.floorMod(hash >>> 9, 8); if (kind < 3) paint.setColor(Color.rgb(8, 30 + base, 40 + base)); else if (kind < 6) paint.setColor(Color.rgb(27 + base, 25 + base, 20 + base / 2)); else paint.setColor(Color.rgb(8 + base / 2, 29 + base, 24 + base / 2));
            canvas.drawRect(sx, sy, sx + tile + 1, sy + tile + 1, paint); paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(1); paint.setColor(Color.argb(50, 95, 210, 190)); canvas.drawRect(sx, sy, sx + tile, sy + tile, paint); paint.setStyle(Paint.Style.FILL);
            if (Math.floorMod(hash >>> 22, 29) == 0) { paint.setColor(Color.rgb(155, 112, 64)); canvas.drawRect(sx + 23, sy + 20, sx + 41, sy + 46, paint); paint.setColor(Color.rgb(205, 159, 83)); canvas.drawCircle(sx + 32, sy + 19, 11, paint); }
        }
        synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f) { float sx = cx + node.x - worldX, sy = cy + node.y - worldY; if (sx > -30 && sy > -30 && sx < width + 30 && sy < height + 30) { int color = node.type == 0 ? Color.rgb(151, 255, 68) : node.type == 1 ? Color.rgb(61, 206, 255) : Color.rgb(255, 190, 66); paint.setColor(Color.argb(90, 0, 0, 0)); canvas.drawOval(sx - 13, sy + 7, sx + 13, sy + 14, paint); paint.setColor(color); float radius = 5 + node.amount * 8; canvas.drawCircle(sx, sy, radius, paint); paint.setColor(Color.argb(210, 235, 255, 235)); canvas.drawCircle(sx - radius * .28f, sy - radius * .28f, Math.max(2f, radius * .24f), paint); } } }
        synchronized (projectiles) { for (Projectile shot : projectiles) {
            float sx = cx + shot.x - worldX, ground = cy + shot.y - worldY; float scale = 1f + shot.z / 260f;
            paint.setColor(Color.argb(90, 0, 0, 0)); canvas.drawOval(sx - 10 * scale, ground - 3, sx + 10 * scale, ground + 4, paint);
            paint.setColor(shot.resting ? Color.rgb(160, 130, 76) : Color.rgb(255, 211, 91)); canvas.drawCircle(sx, ground - shot.z, 7 * scale, paint); paint.setColor(Color.rgb(255, 245, 185)); canvas.drawCircle(sx - 2, ground - shot.z - 2, 2 * scale, paint);
        } }
        float organismSize = Math.min(250f, height * .31f); paint.setColor(Color.argb(115, 0, 0, 0)); canvas.drawOval(cx - organismSize * .34f, cy + organismSize * .38f, cx + organismSize * .34f, cy + organismSize * .52f, paint);
        if (cellularFrame != null) { paint.setFilterBitmap(false); canvas.drawBitmap(cellularFrame, null, new android.graphics.RectF(cx - organismSize * .5f, cy - organismSize * .52f, cx + organismSize * .5f, cy + organismSize * .48f), paint); }
        paint.setColor(Color.argb(150, 255, 90, 180)); paint.setStrokeWidth(3); canvas.drawLine(cx, cy - organismSize * .12f, cx + aimX * 92f, cy - organismSize * .12f + aimY * 92f, paint); canvas.drawCircle(cx + aimX * 92f, cy - organismSize * .12f + aimY * 92f, 5, paint);
        paint.setColor(Color.rgb(67, 239, 220)); paint.setTextSize(25); canvas.drawText("NULLVECTOR // NEURAL HABITAT", 28, 39, paint); paint.setTextSize(15); paint.setColor(Color.rgb(165, 199, 199)); canvas.drawText("CELLULAR CREATURE STAGE · WORLD 4096² · MATERIAL " + gathered, 29, 63, paint);
        bar(canvas, 28, 78, 230, cellularHealth, Color.rgb(62, 224, 115), "HEALTH"); bar(canvas, 28, 102, 230, cellularNeural, Color.rgb(214, 72, 255), "NEURAL");
        paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(3); paint.setColor(movementTouch ? Color.rgb(67, 239, 220) : Color.argb(145, 67, 125, 130)); canvas.drawCircle(width * .13f, height * .82f, 72, paint); paint.setStyle(Paint.Style.FILL); paint.setColor(Color.rgb(67, 239, 220)); canvas.drawCircle(width * .13f + controlX * 58, height * .82f + controlY * 58, 14, paint);
        paint.setColor(actionTouch ? Color.rgb(255, 80, 180) : Color.rgb(84, 45, 73)); canvas.drawCircle(width * .87f, height * .82f, 64, paint); paint.setColor(Color.WHITE); paint.setTextSize(16); canvas.drawText("THROW", width * .87f - 25, height * .82f + 5, paint);
        paint.setColor(Color.argb(180, 5, 10, 14)); canvas.drawRect(width - 145, 18, width - 18, 58, paint); paint.setColor(Color.rgb(140, 205, 205)); paint.setTextSize(14); canvas.drawText(diagnostics ? "HIDE MODELS" : "MODEL INFO", width - 132, 43, paint);
        if (diagnostics) {
            paint.setColor(Color.argb(224, 2, 6, 10)); canvas.drawRect(width * .60f, 72, width - 20, height * .62f, paint); float panelX = width * .60f + 18, panelY = 98;
            paint.setColor(Color.rgb(67, 239, 220)); paint.setTextSize(16); canvas.drawText(status, panelX, panelY, paint); paint.setColor(Color.rgb(170, 195, 202)); paint.setTextSize(14); canvas.drawText(String.format("context %.2fms  action %.2fms  cells %.2fms  VAE %.2fms", milliseconds, actionMilliseconds, cellularMilliseconds, decoderMilliseconds), panelX, panelY + 25, paint);
            if (neuralFrame != null) { float size = Math.min(height * .36f, width * .20f); paint.setFilterBitmap(true); canvas.drawBitmap(neuralFrame, null, new android.graphics.RectF(width - size - 38, panelY + 38, width - 38, panelY + 38 + size), paint); paint.setFilterBitmap(false); }
        }
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        boolean move = false, act = false; float width = Math.max(1, getWidth()), height = Math.max(1, getHeight());
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN && event.getX() > width - 165 && event.getY() < 75) { diagnostics = !diagnostics; invalidate(); return true; }
        if (event.getActionMasked() != MotionEvent.ACTION_UP && event.getActionMasked() != MotionEvent.ACTION_CANCEL) for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            float x = event.getX(pointer), y = event.getY(pointer);
            if (x < width * .55f) { controlX = Math.max(-1, Math.min(1, (x - width * .16f) / 72f)); controlY = Math.max(-1, Math.min(1, (y - height * .82f) / 72f)); move = true; }
            else { float dx = x - width * .5f, dy = y - height * .54f, length = Math.max(1f, (float)Math.hypot(dx, dy)); aimX = dx / length; aimY = dy / length; actionId = 10; if (event.getActionMasked() == MotionEvent.ACTION_DOWN) fireRequests++; act = true; }
        }
        movementTouch = move; actionTouch = act; if (!move) { controlX = 0; controlY = 0; } invalidate(); return true;
    }

    @Override protected void onDetachedFromWindow() { running = false; super.onDetachedFromWindow(); }
}
