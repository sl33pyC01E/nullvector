package world.nullvector.mobile;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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
    private volatile double organismVaeMilliseconds = 0;
    private volatile Bitmap neuralFrame;
    private volatile Bitmap cellularFrame;
    private volatile Bitmap organismVaeFrame;
    private Bitmap neuralTerrain;
    private byte[] neuralTerrainCells;
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
    private volatile int selectedAction = 0;
    private volatile String actionStatus = "SELECT AN ORGANISM";
    private volatile boolean movementTouch = false, actionTouch = false;
    private boolean actionWasDown = false;
    private final List<Projectile> projectiles = new ArrayList<>();
    private final List<MaterialNode> materials = new ArrayList<>();
    private MaterialNode heldMaterial;
    private FoundationWorld foundation;
    private volatile double groundedMilliseconds = 0;
    private volatile double ecologyMilliseconds = 0;

    private static final class Projectile {
        float x, y, z, vx, vy, vz, mass; int bounces, type; boolean resting, impacted;
        Projectile(float x, float y, float aimX, float aimY, int type, float mass) {
            this.x = x; this.y = y; this.z = 54f; this.vx = aimX * 680f; this.vy = aimY * 680f; this.vz = 255f;this.type=type;this.mass=mass;
        }
    }

    private static final class MaterialNode {
        float x, y, amount; final int type;
        MaterialNode(float x, float y, int type, float amount) { this.x = x; this.y = y; this.type = type; this.amount = amount; }
    }

    public NeuralWorldView(Context owner) {
        super(owner); paint.setTypeface(android.graphics.Typeface.MONOSPACE);
        try { foundation = new FoundationWorld(owner); }
        catch (Exception failure) { status = "FOUNDATION LOAD FAILED · " + failure.getMessage(); Log.e(TAG, status, failure); }
        try (InputStream input = owner.getAssets().open("neural_garden_chunk.png")) { neuralTerrain = BitmapFactory.decodeStream(input); } catch (Exception ignored) { neuralTerrain = null; }
        try (InputStream input = owner.getAssets().open("neural_garden_terrain.u8")) { neuralTerrainCells = input.readAllBytes(); if (neuralTerrainCells.length != 1024) neuralTerrainCells = null; } catch (Exception ignored) { neuralTerrainCells = null; }
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

    private Bitmap rgbaBitmap(float[][][][] rgba) {
        int side = 96; int[] pixels = new int[side * side];
        for (int y = 0; y < side; y++) for (int x = 0; x < side; x++) {
            int r = Math.max(0, Math.min(255, Math.round(rgba[0][0][y][x] * 255)));
            int g = Math.max(0, Math.min(255, Math.round(rgba[0][1][y][x] * 255)));
            int b = Math.max(0, Math.min(255, Math.round(rgba[0][2][y][x] * 255)));
            int a = Math.max(0, Math.min(255, Math.round(rgba[0][3][y][x] * 255)));
            pixels[y * side + x] = Color.argb(a, r, g, b);
        }
        return Bitmap.createBitmap(pixels, side, side, Bitmap.Config.ARGB_8888);
    }

    private void bindPhysiologyToRaster(float[] features, float[] anatomy, float[] state) {
        int cells = 48 * 48, cursor = 0;
        for (int pixel = 0; pixel < cells; pixel++) if (anatomy[pixel] > .5f) {
            int offset = cursor * 52; float health = state[pixel], fluid = state[cells + pixel];
            float energy = state[3 * cells + pixel], wound = state[7 * cells + pixel];
            float alive = state[11 * cells + pixel];
            features[offset + 49] = Math.max(0f, Math.min(1f, health * (.65f + .35f * energy)));
            features[offset + 51] = Math.max(0f, Math.min(1f, alive * (1f - .45f * wound) + .08f * fluid));
            cursor++;
        }
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

    private boolean terrainWalkable(float x, float y) {
        if (neuralTerrainCells == null) return true; int cellX = Math.floorMod((int)Math.floor(x / 24f), 32), cellY = Math.floorMod((int)Math.floor(y / 24f), 32);
        if (cellX == 0 || cellY == 0 || cellX == 31 || cellY == 31) return true;
        int value = neuralTerrainCells[cellY * 32 + cellX] & 255; return value == 1 || value == 4 || value == 5 || value == 6 || value == 8;
    }

    private void enforceFoundationTerrain(float[] previous){if(foundation==null)return;synchronized(foundation){for(int i=0;i<foundation.creatures.size();i++){FoundationWorld.Creature c=foundation.creatures.get(i);float radius=28+14*c.traits[0];boolean clear=terrainWalkable(c.x,c.y)&&terrainWalkable(c.x-radius,c.y)&&terrainWalkable(c.x+radius,c.y)&&terrainWalkable(c.x,c.y-radius)&&terrainWalkable(c.x,c.y+radius);if(!clear)foundation.rollbackPosition(i,previous[i*2],previous[i*2+1]);}}}

    private void advanceHabitat(float dt) {
        if (foundation != null) {
            foundation.setPlayerControl(controlX, controlY);
            FoundationWorld.Creature player = foundation.selected();
            worldX = player.x; worldY = player.y; velocityX = player.vx; velocityY = player.vy;
        }
        float targetX = controlX * 250f, targetY = controlY * 205f;
        if (foundation == null) {
        velocityX += (targetX - velocityX) * Math.min(1f, dt * 9f); velocityY += (targetY - velocityY) * Math.min(1f, dt * 9f);
        float proposedX = Math.max(80f, Math.min(4016f, worldX + velocityX * dt)), proposedY = Math.max(80f, Math.min(4016f, worldY + velocityY * dt));
        if (terrainWalkable(proposedX, worldY)) worldX = proposedX; else velocityX *= -.08f; if (terrainWalkable(worldX, proposedY)) worldY = proposedY; else velocityY *= -.08f;
        }
        if(heldMaterial!=null&&heldMaterial.amount>0&&foundation!=null){float[] hand=foundation.selectedGrasperWorld();heldMaterial.x=hand[0];heldMaterial.y=hand[1];}
        if (fireRequests > 0) synchronized (projectiles) { int type=heldMaterial==null?2:heldMaterial.type;float mass=heldMaterial==null?.35f:Math.max(.2f,heldMaterial.amount);projectiles.add(new Projectile(worldX, worldY - 18f, aimX, aimY,type,mass));if(heldMaterial!=null){heldMaterial.amount=0;heldMaterial=null;}fireRequests--; }
        actionWasDown = actionTouch;
        synchronized (projectiles) {
            for (Projectile shot : projectiles) if (!shot.resting) {
                shot.x += shot.vx * dt; shot.y += shot.vy * dt; shot.z += shot.vz * dt; shot.vz -= 560f * dt;
                if (shot.z <= 0f) {
                    shot.z = 0f;
                    if (shot.bounces < 2 && Math.abs(shot.vz) > 80f) { shot.vz = -shot.vz * .34f; shot.vx *= .72f; shot.vy *= .72f; shot.bounces++; }
                    else { shot.vz = 0f; shot.vx *= .88f; shot.vy *= .88f; if (Math.hypot(shot.vx, shot.vy) < 18f) shot.resting = true; }
                }
                if(foundation!=null&&!shot.impacted&&shot.z<28f){float effect=shot.type==0?-.08f*shot.mass:.08f+shot.mass*.12f;int hit=foundation.impact(shot.x,shot.y,22+shot.mass*16,effect);if(hit>0){shot.impacted=true;shot.vx*=-.18f;shot.vy*=-.18f;shot.vz=Math.max(70,shot.vz);}}
                if (shot.z < 34f) synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f && Math.hypot(shot.x - node.x, shot.y - node.y) < 25f) { node.amount = Math.max(0f, node.amount - .42f); gathered++; shot.vx *= -.18f; shot.vy *= -.18f; shot.vz = Math.max(80f, shot.vz); break; } }
            }
            if (projectiles.size() > 48) projectiles.subList(0, projectiles.size() - 48).clear();
        }
        synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f && node.type != 2 && Math.hypot(worldX - node.x, worldY - node.y) < 27f) { float eaten = Math.min(node.amount, dt * .32f); node.amount -= eaten; pendingNutrition += eaten * (node.type == 0 ? 1f : .45f); gathered += node.amount <= 0f ? 1 : 0; } }
    }

    private void performAction(int action){selectedAction=action;if(foundation==null)return;FoundationWorld.Creature player=foundation.selected();if(action==0){MaterialNode best=null;double distance=145; synchronized(materials){for(MaterialNode node:materials)if(node.amount>0){double d=Math.hypot(node.x-player.x,node.y-player.y);if(d<distance){distance=d;best=node;}}}heldMaterial=best;actionStatus=best==null?"GRASP MISSED":"MATERIAL GRASPED";}else if(action==1){if(heldMaterial!=null&&heldMaterial.amount>0){float nutrition=heldMaterial.amount*(heldMaterial.type==0?.34f:heldMaterial.type==1?.12f:.04f);foundation.feedSelected(nutrition);pendingNutrition+=nutrition;heldMaterial.amount=0;heldMaterial=null;actionStatus="FEEDER CONTACT · ABSORBED";}else actionStatus="FEED REQUIRES HELD MATERIAL";}else if(action==2){int hit=foundation.attackSelected(aimX,aimY);actionStatus=hit>0?"CELLULAR STRIKE · "+hit+" CELLS":"STRIKE MISSED";}else if(action==3){int hit=foundation.scrapeSelected(aimX,aimY);actionStatus=hit>0?"SURFACE SCRAPE · "+hit+" CELLS":"SCRAPE MISSED";}else if(action==4){int hit=foundation.cutSelected(aimX,aimY);actionStatus=hit>0?"BOND CUT · "+hit+" CELLS":"CUT MISSED";}else{if(heldMaterial!=null){fireRequests++;actionStatus="BALLISTIC RELEASE";}else actionStatus="THROW REQUIRES HELD MATERIAL";}}

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
                 OrtSession cellularSession = environment.createSession(assetFile("mobile_cell_nca_fp32.onnx").getAbsolutePath(), options);
                 OrtSession organismVaeSession = environment.createSession(assetFile("organism_cell_vae_fp32.onnx").getAbsolutePath(), options);
                 OrtSession groundedSession = environment.createSession(assetFile("grounded_feedback_fp32.onnx").getAbsolutePath(), options);
                 OrtSession ecologySession = environment.createSession(assetFile("mobile_ecology_fp32.onnx").getAbsolutePath(), options)) {
                float[] rawInitial = latentAsset(), latentNorm = floatAsset("latent_normalization.f32", 96), current = new float[rawInitial.length];
                for (int c = 0, i = 0; c < 48; c++) for (int p = 0; p < 32 * 32; p++, i++) current[i] = (rawInitial[i] - latentNorm[c]) / latentNorm[48 + c];
                float[] previous = current.clone(), actor = new float[128], previousActor = new float[128];
                float[] control = new float[4], visibility = new float[32 * 32], memory = new float[32 * 32];
                float[] organismFeatures = floatAsset("organism_vae_features.f32", 576 * 52);
                float[] organismMask = floatAsset("organism_vae_mask.f32", 576);
                java.util.Arrays.fill(visibility, 1f); int tick = 0;
                stage(provider + (BuildConfig.SPLIT_ACTION
                    ? " · INT8 action + cellular NCA + mobile VAE live"
                    : " · FP32 action + cellular NCA + mobile VAE live"));
                  while (running) {
                    long frameBegan = System.nanoTime();
                    if (foundation != null) {
                        FoundationWorld.NeuralBatch batch = foundation.encodeNeural(); long groundedBegan = System.nanoTime();
                        Map<String, OnnxTensor> groundedInputs = new HashMap<>();
                        groundedInputs.put("owner_state", OnnxTensor.createTensor(environment, FloatBuffer.wrap(batch.owner), new long[]{batch.count, FoundationWorld.MAX_APPENDAGES, 23}));
                        groundedInputs.put("global_state", OnnxTensor.createTensor(environment, FloatBuffer.wrap(batch.global), new long[]{batch.count, 23}));
                        groundedInputs.put("owner_mask", OnnxTensor.createTensor(environment, boolMatrix(batch.ownerMask, batch.count, FoundationWorld.MAX_APPENDAGES)));
                        groundedInputs.put("muscle_meta", OnnxTensor.createTensor(environment, FloatBuffer.wrap(batch.muscleMeta), new long[]{batch.count, FoundationWorld.MAX_MUSCLES, 8}));
                        groundedInputs.put("muscle_owner", OnnxTensor.createTensor(environment, longMatrix(batch.muscleOwner, batch.count, FoundationWorld.MAX_MUSCLES)));
                        groundedInputs.put("muscle_mask", OnnxTensor.createTensor(environment, boolMatrix(batch.muscleMask, batch.count, FoundationWorld.MAX_MUSCLES)));
                        try (OrtSession.Result result=groundedSession.run(groundedInputs)) { foundation.applyNeural((float[][])result.get(0).getValue(),(float[][])result.get(1).getValue(),(float[])result.get(2).getValue()); }
                        for(OnnxTensor tensor:groundedInputs.values())tensor.close();groundedMilliseconds=(System.nanoTime()-groundedBegan)/1_000_000.0;float[] previousPositions=foundation.positionSnapshot();foundation.step(1f/30f);enforceFoundationTerrain(previousPositions);
                        if((tick&3)==0){long ecologyBegan=System.nanoTime();for(int ecologyIndex=0;ecologyIndex<foundation.creatures.size();ecologyIndex++){if(ecologyIndex==foundation.selected)continue;FoundationWorld.EcologyInput ecology=foundation.encodeEcology(ecologyIndex);Map<String,OnnxTensor> ecologyInputs=new HashMap<>();ecologyInputs.put("self_features",OnnxTensor.createTensor(environment,FloatBuffer.wrap(ecology.self),new long[]{1,94}));ecologyInputs.put("resource",OnnxTensor.createTensor(environment,FloatBuffer.wrap(ecology.resource),new long[]{1,10,4}));ecologyInputs.put("neighbor",OnnxTensor.createTensor(environment,FloatBuffer.wrap(ecology.neighbor),new long[]{1,12,14}));ecologyInputs.put("neighbor_mask",OnnxTensor.createTensor(environment,FloatBuffer.wrap(ecology.mask),new long[]{1,12}));try(OrtSession.Result result=ecologySession.run(ecologyInputs)){foundation.applyEcology(ecologyIndex,((float[][])result.get(0).getValue())[0],((float[][])result.get(1).getValue())[0],((float[])result.get(2).getValue())[0]);}for(OnnxTensor tensor:ecologyInputs.values())tensor.close();}ecologyMilliseconds=(System.nanoTime()-ecologyBegan)/1_000_000.0;}
                    }
                    advanceHabitat(1f / 30f); control[0] = controlX; control[1] = controlY; control[2] = actionTouch ? 1f : 0f; control[3] = Math.max(-1f, Math.min(1f, cellularHealth * cellularNeural * 2f - 1f));
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
                    if (diagnostics && (neuralFrame == null || tick % 15 == 0)) {
                        float[] rawLatent = new float[current.length]; for (int c = 0, i = 0; c < 48; c++) for (int p = 0; p < 32 * 32; p++, i++) rawLatent[i] = current[i] * latentNorm[48 + c] + latentNorm[c];
                        try (OnnxTensor latentTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(rawLatent), new long[]{1, 48, 32, 32})) {
                            Map<String, OnnxTensor> decoderInput = new HashMap<>(); decoderInput.put("latent", latentTensor); long decoderBegan = System.nanoTime();
                            try (OrtSession.Result result = decoder.run(decoderInput)) { neuralFrame = bitmap((float[][][][])result.get(0).getValue()); }
                            decoderMilliseconds = (System.nanoTime() - decoderBegan) / 1_000_000.0;
                        }
                    }
                    postInvalidateOnAnimation(); tick++;
                    if ((tick & 1) == 0) {
                        long cellularBegan = System.nanoTime();
                        float absorbed = pendingNutrition; pendingNutrition = 0f;
                        if(foundation!=null){if(absorbed>0)foundation.addNutritionToSelected(absorbed);int selectedIndex=foundation.selected;int[] updateIndices=(tick&7)==0?new int[]{selectedIndex,(selectedIndex+1+(tick/8)%4)%5}:new int[]{selectedIndex};for(int physiologyIndex:updateIndices){foundation.preparePhysiology(physiologyIndex);FoundationWorld.Creature body=foundation.creatures.get(physiologyIndex);Map<String,OnnxTensor> cellInputs=new HashMap<>();cellInputs.put("static",OnnxTensor.createTensor(environment,FloatBuffer.wrap(body.cellStatic),new long[]{1,85,48,48}));cellInputs.put("state",OnnxTensor.createTensor(environment,FloatBuffer.wrap(body.cellState),new long[]{1,12,48,48}));cellInputs.put("live_bonds",OnnxTensor.createTensor(environment,FloatBuffer.wrap(body.cellBonds),new long[]{1,8,48,48}));try(OrtSession.Result result=cellularSession.run(cellInputs)){float[][][][] value=(float[][][][])result.get(0).getValue();float[] nextPhysiology=new float[12*48*48];int cursor=0;for(int c=0;c<12;c++)for(int y=0;y<48;y++)for(int x=0;x<48;x++)nextPhysiology[cursor++]=value[0][c][y][x];foundation.applyPhysiology(physiologyIndex,nextPhysiology);}for(OnnxTensor tensor:cellInputs.values())tensor.close();}}
                        cellularMilliseconds = (System.nanoTime() - cellularBegan) / 1_000_000.0;
                        if(foundation!=null){FoundationWorld.Creature body=foundation.selected();cellularHealth=body.health;cellularNeural=body.neural;cellularFrame=cellularBitmap(body.cellStatic,body.cellState);}
                        if (foundation==null && (tick & 7) == 0) {
                            long rasterBegan = System.nanoTime();
                            try (OnnxTensor featureTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(organismFeatures), new long[]{1, 576, 52});
                                 OnnxTensor maskTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(organismMask), new long[]{1, 576})) {
                                Map<String, OnnxTensor> rasterInputs = new HashMap<>(); rasterInputs.put("features", featureTensor); rasterInputs.put("mask", maskTensor);
                                try (OrtSession.Result result = organismVaeSession.run(rasterInputs)) { organismVaeFrame = rgbaBitmap((float[][][][])result.get(0).getValue()); }
                            }
                            organismVaeMilliseconds = (System.nanoTime() - rasterBegan) / 1_000_000.0;
                        }
                    }
                    long remaining = 33_333_333L - (System.nanoTime() - frameBegan); if (remaining > 0) Thread.sleep(remaining / 1_000_000L, (int)(remaining % 1_000_000L));
                  }
            }
        } catch (Throwable failure) {
            Log.e(TAG, "Neural runtime failed", failure);
            status = "SAFE FAILURE · " + failure.getClass().getSimpleName() + " · " + String.valueOf(failure.getMessage());
        }
        postInvalidate();
    }

    private static boolean[][] boolMatrix(boolean[] flat,int rows,int columns){boolean[][] value=new boolean[rows][columns];for(int r=0;r<rows;r++)System.arraycopy(flat,r*columns,value[r],0,columns);return value;}
    private static long[][] longMatrix(long[] flat,int rows,int columns){long[][] value=new long[rows][columns];for(int r=0;r<rows;r++)System.arraycopy(flat,r*columns,value[r],0,columns);return value;}

    private int tissueColor(int tissue,int family){
        int[] colors={0xfff46f7e,0xffeeedcc,0xffff4571,0xfff8bc90,0xff719eb1,0xff48e1f6,0xffea3465,0xff7bd8f4,0xfff1a93f,0xfffdf469,0xffa466e6,0xff73e55c,0xffb94eff,0xffb1c3cf,0xffff833d};
        return colors[Math.floorMod(tissue,colors.length)];
    }

    private void drawFoundationCreatures(Canvas canvas,float cx,float cy,float width,float height){
        if(foundation==null)return; synchronized(foundation){
            for(int index=0;index<foundation.creatures.size();index++){FoundationWorld.Creature creature=foundation.creatures.get(index);float sx=cx+creature.x-worldX,sy=cy+creature.y-worldY;if(sx<-180||sy<-180||sx>width+180||sy>height+180)continue;float scale=creature.selected?4.4f:3.5f;
                paint.setColor(Color.argb(100,0,0,0));canvas.drawOval(sx-28*scale/4,sy+16*scale,sx+28*scale/4,sy+22*scale,paint);
                paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(creature.selected?2.4f:1.2f);paint.setColor(creature.selected?Color.rgb(95,255,222):Color.argb(90,120,220,220));
                for(int edgeIndex=0;edgeIndex<creature.edges.length;edgeIndex++){if(!creature.edgeAlive[edgeIndex])continue;int[] edge=creature.edges[edgeIndex];float ax=sx+(creature.node[edge[0]][0]-creature.bodyProgress)*scale,ay=sy+creature.node[edge[0]][1]*scale,bx=sx+(creature.node[edge[1]][0]-creature.bodyProgress)*scale,by=sy+creature.node[edge[1]][1]*scale;canvas.drawLine(ax,ay,bx,by,paint);}paint.setStyle(Paint.Style.FILL);
                int fluidColor=creature.family==0?Color.rgb(204,45,76):creature.family==1?Color.rgb(123,73,167):creature.family==2?Color.rgb(115,211,66):creature.family==3?Color.rgb(195,77,255):Color.rgb(67,213,239);for(int p=0;p<FoundationWorld.CELL_PIXELS;p++){float surface=creature.cellState[9*FoundationWorld.CELL_PIXELS+p];if(surface>.035f){float puddleX=sx+((p%48)-24)*2.4f,puddleY=sy+22*scale+((p/48)-24)*1.25f;paint.setColor(Color.argb(Math.min(88,(int)(surface*130)),Color.red(fluidColor),Color.green(fluidColor),Color.blue(fluidColor)));canvas.drawCircle(puddleX,puddleY,2.2f+surface*5.5f,paint);}}
                for(FoundationWorld.Cell cell:creature.cells){if(cell.health<=.01f)continue;float px=cell.detached?cx+cell.detachedWorldX-worldX:sx+creature.cellX(cell)*scale,py=cell.detached?cy+cell.detachedWorldY-worldY:sy+creature.cellY(cell)*scale;int alpha=Math.max(35,Math.min(255,(int)(255*cell.alpha*cell.health))),red=Math.min(255,(int)(255*cell.red)),green=Math.min(255,(int)(255*cell.green)),blue=Math.min(255,(int)(255*cell.blue));paint.setColor(Color.argb(alpha,red,green,blue));float radius=(.72f+cell.sigma*.72f)*(creature.selected?2.05f:1.72f);canvas.drawRect(px-radius,py-radius,px+radius,py+radius,paint);}
                paint.setTextSize(12);paint.setColor(creature.selected?Color.rgb(95,255,222):Color.argb(190,190,220,220));String[] intents={"REST","FORAGE","HUNT","FLEE","MATE","FOLLOW","PHOTOSYNTHESIZE","MINE","PHASE FEED","REPAIR","GUARD","EXPLORE"};canvas.drawText(FoundationWorld.FAMILIES[creature.family]+(creature.selected?" // CONTROLLED":" // "+intents[Math.max(0,Math.min(intents.length-1,creature.intent))]),sx-52,sy+34*scale,paint);
            }
        }
    }

    private void bar(Canvas canvas, float x, float y, float width, float value, int color, String label) {
        paint.setColor(Color.argb(180, 5, 11, 16)); canvas.drawRect(x, y, x + width, y + 18, paint); paint.setColor(color); canvas.drawRect(x + 2, y + 2, x + 2 + (width - 4) * Math.max(0f, Math.min(1f, value)), y + 16, paint); paint.setTextSize(13); paint.setColor(Color.WHITE); canvas.drawText(label, x + 6, y + 14, paint);
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); canvas.drawColor(Color.rgb(4, 9, 12)); float width = getWidth(), height = getHeight(); float cx = width * .5f, cy = height * .54f;
        int tile = 64, minX = (int)Math.floor((worldX - cx) / tile) - 1, maxX = (int)Math.ceil((worldX + cx) / tile) + 1;
        int minY = (int)Math.floor((worldY - cy) / tile) - 1, maxY = (int)Math.ceil((worldY + (height - cy)) / tile) + 1;
        if (neuralTerrain != null) { int chunk = 768, chunkMinX = (int)Math.floor((worldX - cx) / chunk) - 1, chunkMaxX = (int)Math.ceil((worldX + cx) / chunk) + 1, chunkMinY = (int)Math.floor((worldY - cy) / chunk) - 1, chunkMaxY = (int)Math.ceil((worldY + height - cy) / chunk) + 1; paint.setFilterBitmap(false); for (int py = chunkMinY; py <= chunkMaxY; py++) for (int px = chunkMinX; px <= chunkMaxX; px++) { float left = cx + px * chunk - worldX, top = cy + py * chunk - worldY; canvas.drawBitmap(neuralTerrain, null, new android.graphics.RectF(left, top, left + chunk, top + chunk), paint); } }
        for (int ty = minY; ty <= maxY; ty++) for (int tx = minX; tx <= maxX; tx++) {
            long hash = worldHash(tx, ty); int kind = (int)Math.floorMod(hash, 17); float sx = cx + tx * tile - worldX, sy = cy + ty * tile - worldY;
            if (neuralTerrain == null) { int base = 13 + (int)Math.floorMod(hash >>> 9, 8); if (kind < 3) paint.setColor(Color.rgb(8, 30 + base, 40 + base)); else if (kind < 6) paint.setColor(Color.rgb(27 + base, 25 + base, 20 + base / 2)); else paint.setColor(Color.rgb(8 + base / 2, 29 + base, 24 + base / 2)); canvas.drawRect(sx, sy, sx + tile + 1, sy + tile + 1, paint); }
            paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(1); paint.setColor(Color.argb(32, 95, 210, 190)); canvas.drawRect(sx, sy, sx + tile, sy + tile, paint); paint.setStyle(Paint.Style.FILL);
            if (Math.floorMod(hash >>> 22, 29) == 0) { paint.setColor(Color.rgb(155, 112, 64)); canvas.drawRect(sx + 23, sy + 20, sx + 41, sy + 46, paint); paint.setColor(Color.rgb(205, 159, 83)); canvas.drawCircle(sx + 32, sy + 19, 11, paint); }
        }
        synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f) { float sx = cx + node.x - worldX, sy = cy + node.y - worldY; if (sx > -30 && sy > -30 && sx < width + 30 && sy < height + 30) { int color = node.type == 0 ? Color.rgb(151, 255, 68) : node.type == 1 ? Color.rgb(61, 206, 255) : Color.rgb(255, 190, 66); paint.setColor(Color.argb(90, 0, 0, 0)); canvas.drawOval(sx - 13, sy + 7, sx + 13, sy + 14, paint); paint.setColor(color); float radius = 5 + node.amount * 8; canvas.drawCircle(sx, sy, radius, paint); paint.setColor(Color.argb(210, 235, 255, 235)); canvas.drawCircle(sx - radius * .28f, sy - radius * .28f, Math.max(2f, radius * .24f), paint); } } }
        synchronized (projectiles) { for (Projectile shot : projectiles) {
            float sx = cx + shot.x - worldX, ground = cy + shot.y - worldY; float scale = 1f + shot.z / 260f;
            paint.setColor(Color.argb(90, 0, 0, 0)); canvas.drawOval(sx - 10 * scale, ground - 3, sx + 10 * scale, ground + 4, paint);
            paint.setColor(shot.resting ? Color.rgb(160, 130, 76) : Color.rgb(255, 211, 91)); canvas.drawCircle(sx, ground - shot.z, 7 * scale, paint); paint.setColor(Color.rgb(255, 245, 185)); canvas.drawCircle(sx - 2, ground - shot.z - 2, 2 * scale, paint);
        } }
        drawFoundationCreatures(canvas,cx,cy,width,height);
        float organismSize = Math.min(250f, height * .31f); paint.setColor(Color.argb(115, 0, 0, 0));
        if(foundation==null) canvas.drawOval(cx - organismSize * .34f, cy + organismSize * .38f, cx + organismSize * .34f, cy + organismSize * .52f, paint);
        android.graphics.RectF organismRect = new android.graphics.RectF(cx - organismSize * .5f, cy - organismSize * .52f, cx + organismSize * .5f, cy + organismSize * .48f);
        paint.setFilterBitmap(false); paint.setAlpha(255);
        if (foundation==null && organismVaeFrame != null) canvas.drawBitmap(organismVaeFrame, null, organismRect, paint);
        if (foundation==null && cellularFrame != null) { paint.setAlpha(organismVaeFrame == null ? 255 : 76); canvas.drawBitmap(cellularFrame, null, organismRect, paint); paint.setAlpha(255); }
        paint.setColor(Color.argb(150, 255, 90, 180)); paint.setStrokeWidth(3); canvas.drawLine(cx, cy - organismSize * .12f, cx + aimX * 92f, cy - organismSize * .12f + aimY * 92f, paint); canvas.drawCircle(cx + aimX * 92f, cy - organismSize * .12f + aimY * 92f, 5, paint);
        paint.setColor(Color.rgb(67, 239, 220)); paint.setTextSize(25); canvas.drawText("NULLVECTOR // NEURAL HABITAT", 28, 39, paint); paint.setTextSize(15); paint.setColor(Color.rgb(165, 199, 199)); canvas.drawText("CELLULAR CREATURE STAGE · WORLD 4096² · MATERIAL " + gathered, 29, 63, paint);
        FoundationWorld.Creature selectedBody=foundation==null?null:foundation.selected();
        bar(canvas,28,78,158,cellularHealth,Color.rgb(62,224,115),"HEALTH");bar(canvas,28,102,158,cellularNeural,Color.rgb(214,72,255),"NEURAL");
        if(selectedBody!=null){bar(canvas,198,78,142,selectedBody.circulation,Color.rgb(245,77,101),"CIRCULATION");bar(canvas,198,102,142,selectedBody.respiration,Color.rgb(71,213,242),"RESPIRATION");bar(canvas,352,78,142,selectedBody.digestion,Color.rgb(245,174,62),"DIGESTION");bar(canvas,352,102,142,selectedBody.locomotion,Color.rgb(155,240,80),"LOCOMOTION");bar(canvas,506,78,142,selectedBody.sensory,Color.rgb(207,118,255),"SENSORY");bar(canvas,506,102,142,selectedBody.energy,Color.rgb(255,225,92),"ENERGY");}
        if(foundation!=null){float cardWidth=Math.min(112,(width-40)/5);for(int i=0;i<5;i++){float left=20+i*cardWidth;paint.setColor(i==foundation.selected?Color.argb(220,16,75,75):Color.argb(190,5,17,22));canvas.drawRect(left,130,left+cardWidth-5,166,paint);paint.setColor(i==foundation.selected?Color.rgb(105,255,220):Color.rgb(160,190,200));paint.setTextSize(11);canvas.drawText(FoundationWorld.FAMILIES[i],left+5,152,paint);}}
        paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(3); paint.setColor(movementTouch ? Color.rgb(67, 239, 220) : Color.argb(145, 67, 125, 130)); canvas.drawCircle(width * .13f, height * .82f, 72, paint); paint.setStyle(Paint.Style.FILL); paint.setColor(Color.rgb(67, 239, 220)); canvas.drawCircle(width * .13f + controlX * 58, height * .82f + controlY * 58, 14, paint);
        String[] actions={"GRASP","FEED","STRIKE","SCRAPE","CUT","THROW"};float actionStart=width*.54f,actionGap=Math.min(105,width*.073f),actionY=height*.84f;for(int i=0;i<actions.length;i++){float x=actionStart+i*actionGap;paint.setColor(i==selectedAction?Color.rgb(132,42,103):Color.rgb(48,30,47));canvas.drawCircle(x,actionY,39,paint);paint.setColor(Color.WHITE);paint.setTextSize(11);canvas.drawText(actions[i],x-20,actionY+4,paint);}paint.setColor(Color.rgb(170,205,205));paint.setTextSize(12);canvas.drawText(actionStatus,actionStart,actionY+62,paint);
        paint.setColor(Color.argb(180, 5, 10, 14)); canvas.drawRect(width - 145, 18, width - 18, 58, paint); paint.setColor(Color.rgb(140, 205, 205)); paint.setTextSize(14); canvas.drawText(diagnostics ? "HIDE MODELS" : "MODEL INFO", width - 132, 43, paint);
        if (diagnostics) {
            float left = width * .47f, panelX = left + 18, panelY = 100; paint.setColor(Color.argb(238, 2, 6, 10)); canvas.drawRect(left, 72, width - 20, height * .73f, paint);
            paint.setColor(Color.rgb(67, 239, 220)); paint.setTextSize(17); canvas.drawText("NEURAL DEBUG // INTERNAL RUNTIME", panelX, panelY, paint);
            paint.setColor(Color.rgb(160, 190, 200)); paint.setTextSize(13); canvas.drawText(status, panelX, panelY + 22, paint);
            canvas.drawText(String.format("WORLD CONTEXT ENCODER · FP32 · STARTUP %.2f ms", milliseconds), panelX, panelY + 48, paint);
            canvas.drawText(String.format("RECURRENT ACTION CORE · %s · 30 Hz · %.2f ms", BuildConfig.SPLIT_ACTION ? "INT8" : "FP32", actionMilliseconds), panelX, panelY + 68, paint);
            canvas.drawText(String.format("CELL PHYSIOLOGY NCA · 492,492 PARAM · 15 Hz · %.2f ms", cellularMilliseconds), panelX, panelY + 88, paint);
            canvas.drawText(String.format("GROUNDED MUSCLE/CONTACT POLICY · 3.50M PARAM · 30 Hz · %.2f ms", groundedMilliseconds), panelX, panelY + 108, paint);
            canvas.drawText(String.format("ECOLOGY INTENT/STEERING POLICY · 125,127 PARAM · 7.5 Hz · %.2f ms", ecologyMilliseconds), panelX, panelY + 128, paint);
            canvas.drawText(String.format("ORGANISM CELL VAE · 138,539 PARAM · 3.75 Hz · %.2f ms", organismVaeMilliseconds), panelX, panelY + 148, paint);
            canvas.drawText(String.format("WORLD FRAME VAE · 91,407 PARAM · DEBUG 2 Hz · %.2f ms", decoderMilliseconds), panelX, panelY + 168, paint);
            if (neuralFrame != null) {
                float size = Math.min(height * .28f, width * .17f), frameLeft = width - size - 38, frameTop = panelY + 148;
                paint.setFilterBitmap(true); canvas.drawBitmap(neuralFrame, null, new android.graphics.RectF(frameLeft, frameTop, frameLeft + size, frameTop + size), paint); paint.setFilterBitmap(false);
                paint.setColor(Color.rgb(255, 92, 188)); paint.setTextSize(12); canvas.drawText("WORLD-LATENT DECODER PROBE", frameLeft, frameTop - 8, paint);
                paint.setColor(Color.rgb(150, 175, 185)); canvas.drawText("not the creature raster", frameLeft, frameTop + size + 16, paint);
            }
            float orbitX = panelX + 92, orbitY = panelY + 220; paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(1); paint.setColor(Color.argb(100, 67, 239, 220)); canvas.drawCircle(orbitX, orbitY, 68, paint); paint.setStyle(Paint.Style.FILL);
            for (int i = 0; i < Math.min(32, context.length); i++) { double angle = i * Math.PI * 2 / 32; float radius = 28 + Math.abs(context[i]) * 40; paint.setColor(Color.argb(180, 67, 239, 220)); canvas.drawCircle(orbitX + (float)Math.cos(angle) * radius, orbitY + (float)Math.sin(angle) * radius, 2 + Math.min(5, Math.abs(context[i]) * 4), paint); }
            paint.setColor(Color.rgb(150, 175, 185)); paint.setTextSize(12); canvas.drawText("64D WORLD CONTEXT", orbitX - 64, orbitY + 90, paint);
        }
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        boolean move = false, act = false; float width = Math.max(1, getWidth()), height = Math.max(1, getHeight());
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN && event.getX() > width - 165 && event.getY() < 75) { diagnostics = !diagnostics; invalidate(); return true; }
        if(event.getActionMasked()==MotionEvent.ACTION_DOWN&&foundation!=null&&event.getY()>=126&&event.getY()<=174){int index=(int)((event.getX()-20)/Math.min(112,(width-40)/5));if(index>=0&&index<5){foundation.select(index);FoundationWorld.Creature selected=foundation.selected();worldX=selected.x;worldY=selected.y;invalidate();return true;}}
        if(event.getActionMasked()==MotionEvent.ACTION_DOWN&&event.getY()>height*.76f&&event.getX()>width*.49f){float gap=Math.min(105,width*.073f),start=width*.54f;int index=Math.round((event.getX()-start)/gap);if(index>=0&&index<6){performAction(index);invalidate();return true;}}
        if (event.getActionMasked() != MotionEvent.ACTION_UP && event.getActionMasked() != MotionEvent.ACTION_CANCEL) for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            float x = event.getX(pointer), y = event.getY(pointer);
            if (x < width * .55f) { controlX = Math.max(-1, Math.min(1, (x - width * .16f) / 72f)); controlY = Math.max(-1, Math.min(1, (y - height * .82f) / 72f)); move = true; }
            else { float dx = x - width * .5f, dy = y - height * .54f, length = Math.max(1f, (float)Math.hypot(dx, dy)); aimX = dx / length; aimY = dy / length; actionId = 10; act = true; }
        }
        movementTouch = move; actionTouch = act; if (!move) { controlX = 0; controlY = 0; } invalidate(); return true;
    }

    @Override protected void onDetachedFromWindow() { running = false; super.onDetachedFromWindow(); }
}
