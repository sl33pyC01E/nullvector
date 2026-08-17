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
import android.graphics.Path;
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
import java.util.Locale;

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
    private volatile boolean hudVisible = true, sightOverlay = true, labelsVisible = true, barsVisible = true;
    private final boolean[] exploredWorld = new boolean[128 * 128];
    private volatile int gathered = 0;
    private volatile float pendingNutrition = 0f;
    private volatile int actionId = 0;
    private volatile int selectedAction = 0;
    private volatile String actionStatus = "SELECT AN ORGANISM";
    private volatile boolean movementTouch = false, actionTouch = false;
    private volatile boolean gameStarted = false, godMode = false;
    private volatile int setupFamily = 0, setupMode = 0;
    private volatile float setupSpeed = .55f, setupSensory = .55f, setupResilience = .55f;
    private int movementPointer = -1, aimPointer = -1;
    private final List<Projectile> projectiles = new ArrayList<>();
    private final List<MaterialNode> materials = new ArrayList<>();
    private MaterialNode heldMaterial;
    private FoundationWorld foundation;
    private volatile double groundedMilliseconds = 0;
    private volatile double ecologyMilliseconds = 0;
    private volatile double grasperMilliseconds = 0;
    private volatile double macroMilliseconds = 0, colonyMilliseconds = 0, societyMilliseconds = 0, timelineMilliseconds = 0, counterfactualMilliseconds = 0;
    private volatile int interactionGoal = -1, grasperOwner = -1;
    private volatile float interactionAge = 0f;
    private MaterialNode interactionTarget;

    private static final class Projectile {
        float x, y, z, vx, vy, vz, mass, damage; int bounces, type, ability; boolean resting, impacted;
        Projectile(float x, float y, float aimX, float aimY, int type, float mass) {
            this.x = x; this.y = y; this.z = 54f; this.vx = aimX * 680f; this.vy = aimY * 680f; this.vz = 255f;this.type=type;this.mass=mass;
        }
        static Projectile ability(float x,float y,float aimX,float aimY,int ability){Projectile shot=new Projectile(x,y,aimX,aimY,2,ability==1?.42f:.28f);shot.ability=ability;shot.damage=ability==1?.24f:ability==2?.17f:.20f;shot.z=ability==2?38f:48f;shot.vx=aimX*(ability==1?980f:ability==2?610f:790f);shot.vy=aimY*(ability==1?980f:ability==2?610f:790f);shot.vz=ability==1?92f:ability==2?18f:70f;return shot;}
    }

    private static final class MaterialNode {
        float x, y, amount; final int type; final FoundationWorld.Creature creature; final FoundationWorld.Cell cell;
        MaterialNode(float x,float y,int type,float amount){this(x,y,type,amount,null,null);}
        MaterialNode(float x,float y,int type,float amount,FoundationWorld.Creature creature,FoundationWorld.Cell cell){this.x=x;this.y=y;this.type=type;this.amount=amount;this.creature=creature;this.cell=cell;}
        boolean isCreature(){return creature!=null&&cell==null;} boolean isFragment(){return cell!=null;}
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

    private String ensembleModel(String stem){return stem+(BuildConfig.SPLIT_ACTION?"_int8.onnx":"_fp32.onnx");}

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

    private static float wrappedDelta(float from,float to){float value=to-from;if(value>2048f)value-=4096f;else if(value<-2048f)value+=4096f;return value;}

    private boolean visibleOffset(float dx,float dy,float sensory){float distance=(float)Math.hypot(dx,dy),hearing=150f+sensory*150f;if(distance<=hearing)return true;float vision=430f+sensory*470f;if(distance>vision)return false;float inverse=1f/Math.max(1f,distance),dot=(dx*aimX+dy*aimY)*inverse;return dot>=.48f;}

    private boolean isWorldVisible(float x,float y){if(foundation==null)return true;FoundationWorld.Creature body=foundation.selected();return visibleOffset(wrappedDelta(body.x,x),wrappedDelta(body.y,y),body.sensory);}

    private int exploredIndex(float x,float y){int gx=Math.floorMod((int)Math.floor(x/32f),128),gy=Math.floorMod((int)Math.floor(y/32f),128);return gy*128+gx;}

    private boolean isWorldRemembered(float x,float y){return exploredWorld[exploredIndex(x,y)];}

    private void updatePerception(float[] visibility,float[] memory){if(foundation==null){java.util.Arrays.fill(visibility,1f);java.util.Arrays.fill(memory,1f);return;}FoundationWorld.Creature body=foundation.selected();for(int y=0;y<32;y++)for(int x=0;x<32;x++){int index=y*32+x;float dx=(x-15.5f)*32f,dy=(y-15.5f)*32f;boolean seen=visibleOffset(dx,dy,body.sensory);visibility[index]=seen?1f:0f;int worldIndex=exploredIndex(body.x+dx,body.y+dy);if(seen)exploredWorld[worldIndex]=true;memory[index]=exploredWorld[worldIndex]?1f:0f;}}

    private void tickGrasper(OrtEnvironment environment,OrtSession session)throws Exception {
        MaterialNode target=interactionTarget;
        if(foundation==null||target==null||target.amount<=0||interactionGoal<0)return;
        FoundationWorld.Creature body=foundation.selected();
        float queryX=interactionGoal==4?body.x+aimX*140:target.x,queryY=interactionGoal==4?body.y+aimY*140:target.y;
        FoundationWorld.GrasperInput input=foundation.encodeGrasper(queryX,queryY,target.amount,interactionGoal,heldMaterial==target);
        Map<String,OnnxTensor> tensors=new HashMap<>();
        tensors.put("owner_meta",OnnxTensor.createTensor(environment,FloatBuffer.wrap(input.owner),new long[]{1,FoundationWorld.MAX_APPENDAGES,16}));
        tensors.put("owner_mask",OnnxTensor.createTensor(environment,boolMatrix(input.mask,1,FoundationWorld.MAX_APPENDAGES)));
        tensors.put("target",OnnxTensor.createTensor(environment,FloatBuffer.wrap(input.target),new long[]{1,18}));
        tensors.put("global_state",OnnxTensor.createTensor(environment,FloatBuffer.wrap(input.global),new long[]{1,10}));
        long began=System.nanoTime();FoundationWorld.GrasperCommand command;
        try(OrtSession.Result result=session.run(tensors)){command=foundation.applyGrasper(((float[][])result.get(0).getValue())[0],((float[])result.get(1).getValue())[0],((float[][])result.get(2).getValue())[0],((float[])result.get(3).getValue())[0],((float[])result.get(6).getValue())[0],((float[][])result.get(7).getValue())[0]);}
        finally{for(OnnxTensor tensor:tensors.values())tensor.close();}
        grasperMilliseconds=(System.nanoTime()-began)/1_000_000.0;grasperOwner=command.owner;
        float[] hand=foundation.selectedGrasperWorld(command.owner);float distance=(float)Math.hypot(hand[0]-target.x,hand[1]-target.y);
        boolean physicalClosure=interactionAge>1.1f&&distance<84f;
        if(heldMaterial==null&&((command.engage&&distance<28)||physicalClosure)){
            heldMaterial=target;actionStatus=target.isCreature()?"ENTITY GRIP CLOSED":target.isFragment()?"TISSUE GRIP CLOSED":"PHYSICAL GRIP CLOSED";Log.i(TAG,actionStatus);
        }
        if(heldMaterial!=target)return;
        target.x=hand[0];target.y=hand[1];
        if(target.isCreature())foundation.carryCreature(target.creature,target.x,target.y);
        else if(target.isFragment()){target.cell.detached=true;target.cell.detachedWorldX=target.x;target.cell.detachedWorldY=target.y;}
        if(interactionGoal==1){
            float[] feeder=foundation.selectedFeederWorld();float contact=(float)Math.hypot(target.x-feeder[0],target.y-feeder[1]);
            if(contact<19){float absorbed=Math.min(target.amount,.012f*Math.max(.15f,body.digestion));target.amount-=absorbed;if(target.isFragment())target.cell.health=Math.max(0,target.amount);pendingNutrition+=absorbed*(target.type==0?.34f:target.type==1?.12f:.04f);actionStatus="LIVE FEEDER CONTACT · "+Math.round((1-target.amount)*100)+"%";if(target.amount<=.001f){target.amount=0;heldMaterial=null;interactionTarget=null;interactionGoal=-1;}}
        }else if(interactionGoal==4&&(command.release||interactionAge>.55f)){
            float dx=aimX,dy=aimY,length=Math.max(.001f,(float)Math.hypot(dx,dy));dx/=length;dy/=length;
            if(target.isCreature())foundation.throwCreature(target.creature,dx,dy,720f);
            else{synchronized(projectiles){projectiles.add(new Projectile(target.x,target.y,dx,dy,target.type,Math.max(.2f,target.amount)));}if(target.isFragment())target.cell.health=0;target.amount=0;}
            heldMaterial=null;interactionTarget=null;interactionGoal=-1;actionStatus="AIMED BALLISTIC RELEASE";Log.i(TAG,actionStatus);
        }
    }

    private boolean terrainWalkable(float x, float y) {
        if (neuralTerrainCells == null) return true; int cellX = Math.floorMod((int)Math.floor(x / 24f), 32), cellY = Math.floorMod((int)Math.floor(y / 24f), 32);
        if(foundation!=null&&foundation.structureBlocked(x,y))return false;if (cellX == 0 || cellY == 0 || cellX == 31 || cellY == 31) return true;
        int value = neuralTerrainCells[cellY * 32 + cellX] & 255; return value == 1 || value == 4 || value == 5 || value == 6 || value == 8;
    }

    private void enforceFoundationTerrain(float[] previous){if(foundation==null)return;synchronized(foundation){for(int i=0;i<foundation.creatures.size();i++){FoundationWorld.Creature c=foundation.creatures.get(i);float radius=28+14*c.traits[0];boolean clear=terrainWalkable(c.x,c.y)&&terrainWalkable(c.x-radius,c.y)&&terrainWalkable(c.x+radius,c.y)&&terrainWalkable(c.x,c.y-radius)&&terrainWalkable(c.x,c.y+radius);if(!clear)foundation.rollbackPosition(i,previous[i*2],previous[i*2+1]);}}}

    private void advanceHabitat(float dt) {
        if(interactionGoal>=0)interactionAge+=dt;
        if (foundation != null) {
            FoundationWorld.Creature player = foundation.selected();
            float physicalControlX=controlX,physicalControlY=controlY;
            if(!movementTouch&&interactionTarget!=null&&heldMaterial==null&&interactionGoal>=0&&interactionGoal!=4){float dx=interactionTarget.x-player.x,dy=interactionTarget.y-player.y,distance=(float)Math.hypot(dx,dy);if(distance>78f){physicalControlX=dx/distance*.62f;physicalControlY=dy/distance*.62f;}}
            foundation.setPlayerControl(physicalControlX, physicalControlY);
            worldX = player.x; worldY = player.y; velocityX = player.vx; velocityY = player.vy;
        }
        float targetX = controlX * 250f, targetY = controlY * 205f;
        if (foundation == null) {
        velocityX += (targetX - velocityX) * Math.min(1f, dt * 9f); velocityY += (targetY - velocityY) * Math.min(1f, dt * 9f);
        float proposedX = Math.max(80f, Math.min(4016f, worldX + velocityX * dt)), proposedY = Math.max(80f, Math.min(4016f, worldY + velocityY * dt));
        if (terrainWalkable(proposedX, worldY)) worldX = proposedX; else velocityX *= -.08f; if (terrainWalkable(worldX, proposedY)) worldY = proposedY; else velocityY *= -.08f;
        }
        syncGrabbables();if(heldMaterial!=null&&heldMaterial.amount>0&&foundation!=null){float[] hand=foundation.selectedGrasperWorld(grasperOwner);heldMaterial.x=hand[0];heldMaterial.y=hand[1];}
        synchronized (projectiles) {
            for (Projectile shot : projectiles) if (!shot.resting) {
                shot.x += shot.vx * dt; shot.y += shot.vy * dt; shot.z += shot.vz * dt; shot.vz -= 560f * dt;
                if (shot.z <= 0f) {
                    shot.z = 0f;
                    if (shot.bounces < 2 && Math.abs(shot.vz) > 80f) { shot.vz = -shot.vz * .34f; shot.vx *= .72f; shot.vy *= .72f; shot.bounces++; }
                    else { shot.vz = 0f; shot.vx *= .88f; shot.vy *= .88f; if (Math.hypot(shot.vx, shot.vy) < 18f) shot.resting = true; }
                }
                if(foundation!=null&&!shot.impacted&&shot.z<34f){float effect=shot.ability>0?shot.damage:(shot.type==0?-.08f*shot.mass:.08f+shot.mass*.12f);int hit=foundation.impact(shot.x,shot.y,(shot.ability==2?46:22)+shot.mass*16,effect);if(hit>0){shot.impacted=true;shot.vx*=-.18f;shot.vy*=-.18f;shot.vz=Math.max(shot.ability==2?18:70,shot.vz);}}
                if (shot.z < 34f) synchronized (materials) { for (MaterialNode node : materials) if (node.amount > 0f && Math.hypot(shot.x - node.x, shot.y - node.y) < 25f) { node.amount = Math.max(0f, node.amount - .42f); gathered++; shot.vx *= -.18f; shot.vy *= -.18f; shot.vz = Math.max(80f, shot.vz); break; } }
            }
            if (projectiles.size() > 48) projectiles.subList(0, projectiles.size() - 48).clear();
        }
    }

    private void syncGrabbables(){if(foundation==null)return;synchronized(foundation){synchronized(materials){for(MaterialNode node:materials){if(node==heldMaterial)continue;if(node.isCreature()){node.x=node.creature.x;node.y=node.creature.y;node.amount=.8f+node.creature.traits[0]*1.4f;}else if(node.isFragment()){node.x=node.cell.detached?node.cell.detachedWorldX:node.creature.x+node.creature.cellX(node.cell)*4f;node.y=node.cell.detached?node.cell.detachedWorldY:node.creature.y+node.creature.cellY(node.cell)*4f;node.amount=node.cell.health;}}}}}

    private MaterialNode nearestMaterial(float maxDistance){if(foundation==null)return null;MaterialNode best=null;double distance=maxDistance;synchronized(foundation){FoundationWorld.Creature player=foundation.selected();synchronized(materials){for(MaterialNode node:materials)if(node.amount>0){double d=Math.hypot(node.x-player.x,node.y-player.y);if(d<distance){distance=d;best=node;}}for(FoundationWorld.Creature creature:foundation.creatures){if(creature==player)continue;double bodyDistance=Math.hypot(creature.x-player.x,creature.y-player.y);if(bodyDistance<distance){MaterialNode existing=null;for(MaterialNode node:materials)if(node.creature==creature&&node.cell==null){existing=node;break;}if(existing==null){existing=new MaterialNode(creature.x,creature.y,0,.8f+creature.traits[0]*1.4f,creature,null);materials.add(existing);}best=existing;distance=bodyDistance;}for(FoundationWorld.Cell cell:creature.cells)if(cell.detached||(creature.health<.18f&&cell.health>.02f)){float x=cell.detached?cell.detachedWorldX:creature.x+creature.cellX(cell)*4f,y=cell.detached?cell.detachedWorldY:creature.y+creature.cellY(cell)*4f,d=(float)Math.hypot(x-player.x,y-player.y);if(d<distance){MaterialNode existing=null;for(MaterialNode node:materials)if(node.cell==cell){existing=node;break;}if(existing==null){existing=new MaterialNode(x,y,0,cell.health,creature,cell);materials.add(existing);}best=existing;distance=d;}}}}}return best;}

    private void performAction(int action){selectedAction=action;if(foundation==null)return;float physicalWorkspace=240f;if(action==0){interactionTarget=nearestMaterial(physicalWorkspace);interactionGoal=2;actionStatus=interactionTarget==null?"NO GRASPABLE TARGET IN WORKSPACE":interactionTarget.isCreature()?"REACHING FOR ENTITY":interactionTarget.isFragment()?"REACHING FOR CORPSE FRAGMENT":"NEURAL GRASPER REACHING";}else if(action==1){interactionTarget=heldMaterial!=null?heldMaterial:nearestMaterial(physicalWorkspace);interactionGoal=1;actionStatus=interactionTarget==null?"NO FEEDSTOCK IN PHYSICAL WORKSPACE":"ROUTING TO LIVE FEEDER";}else if(action==2){interactionGoal=-1;int hit=foundation.attackSelected(aimX,aimY);actionStatus=hit>0?"CELLULAR STRIKE · "+hit+" CELLS":"STRIKE MISSED";}else if(action==3){interactionGoal=-1;int hit=foundation.scrapeSelected(aimX,aimY);actionStatus=hit>0?"SURFACE SCRAPE · "+hit+" CELLS":"SCRAPE MISSED";}else if(action==4){interactionGoal=-1;int hit=foundation.cutSelected(aimX,aimY);actionStatus=hit>0?"BOND CUT · "+hit+" CELLS":"CUT MISSED";}else if(action==5){if(heldMaterial!=null){interactionTarget=heldMaterial;interactionGoal=4;actionStatus="THROW ARMED · AIM WITH RIGHT STICK";}else{interactionGoal=-1;interactionTarget=null;actionStatus="THROW LOCKED · GRASP SOMETHING FIRST";}}else{interactionGoal=-1;int ability=foundation.selectedProjectileAbility();if(ability==0)actionStatus="NO PROJECTILE ORGAN OR MODULE";else if(!foundation.consumeSelectedProjectileCost())actionStatus="PROJECTILE CHARGING OR LOW ENERGY";else{float length=Math.max(.001f,(float)Math.hypot(aimX,aimY)),dx=aimX/length,dy=aimY/length;FoundationWorld.Creature body=foundation.selected();synchronized(projectiles){projectiles.add(Projectile.ability(body.x+dx*46,body.y+dy*46,dx,dy,ability));}actionStatus=ability==1?"MACHINE KINETIC FIRED":ability==2?"ANOMALY PHASE BOLT FIRED":"GRAFTED EMITTER FIRED";}}}

    private void runModels() {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            OrtEnvironment environment = OrtEnvironment.getEnvironment();
            // The first preview enabled NNAPI generically. On current Samsung
            // firmware that can take the whole process down inside the vendor
            // driver before Java receives an exception. Keep this recovery
            // build on ORT CPU until a model-by-model QNN partition is tested.
            options.setIntraOpNumThreads(2); options.setInterOpNumThreads(1);
            String provider = "ORT CPU SAFE";
            stage("Loading coupled neural world ensemble…");
            String actionModel = BuildConfig.SPLIT_ACTION ? "action_delta_int8_qdq.onnx" : "action_core_fp32.onnx";
            stage(provider + " · loading " + (BuildConfig.SPLIT_ACTION ? "INT8" : "FP32") + " action runtime…");
            try (OrtSession contextSession = environment.createSession(assetFile("world_context_fp32.onnx").getAbsolutePath(), options);
                 OrtSession actionSession = environment.createSession(assetFile(actionModel).getAbsolutePath(), options);
                 OrtSession actorSession = BuildConfig.SPLIT_ACTION ? environment.createSession(assetFile("actor_state_fp32.onnx").getAbsolutePath(), options) : null;
                 OrtSession decoder = environment.createSession(assetFile("frame_vae_fp32.onnx").getAbsolutePath(), options);
                 OrtSession cellularSession = environment.createSession(assetFile("mobile_cell_nca_fp32.onnx").getAbsolutePath(), options);
                 OrtSession organismVaeSession = environment.createSession(assetFile("organism_cell_vae_fp32.onnx").getAbsolutePath(), options);
                 OrtSession groundedSession = environment.createSession(assetFile("grounded_feedback_fp32.onnx").getAbsolutePath(), options);
                 OrtSession ecologySession = environment.createSession(assetFile("mobile_ecology_fp32.onnx").getAbsolutePath(), options);
                 OrtSession grasperSession = environment.createSession(assetFile("neural_grasper_fp32.onnx").getAbsolutePath(), options);
                 OrtSession macroSession = environment.createSession(assetFile(ensembleModel("macro")).getAbsolutePath(), options);
                 OrtSession colonySession = environment.createSession(assetFile(ensembleModel("colony")).getAbsolutePath(), options);
                 OrtSession societySession = environment.createSession(assetFile(ensembleModel("society")).getAbsolutePath(), options);
                 OrtSession timelineSession = environment.createSession(assetFile(ensembleModel("timeline")).getAbsolutePath(), options);
                 OrtSession counterfactualSession = environment.createSession(assetFile(ensembleModel("counterfactual")).getAbsolutePath(), options)) {
                float[] rawInitial = latentAsset(), latentNorm = floatAsset("latent_normalization.f32", 96), current = new float[rawInitial.length];
                for (int c = 0, i = 0; c < 48; c++) for (int p = 0; p < 32 * 32; p++, i++) current[i] = (rawInitial[i] - latentNorm[c]) / latentNorm[48 + c];
                float[] previous = current.clone(), actor = new float[128], previousActor = new float[128];
                float[] control = new float[4], visibility = new float[32 * 32], memory = new float[32 * 32];
                float[] organismFeatures = floatAsset("organism_vae_features.f32", 576 * 52);
                float[] organismMask = floatAsset("organism_vae_mask.f32", 576);
                int tick = 0;
                stage(provider + (BuildConfig.SPLIT_ACTION
                    ? " · INT8 ensemble live · SCAFFOLD CELL RASTER"
                    : " · FP32 ensemble live · SCAFFOLD CELL RASTER"));
                  while (running) {
                    long frameBegan = System.nanoTime();
                    if(!gameStarted){postInvalidateOnAnimation();Thread.sleep(33);continue;}
                    if (foundation != null) {
                        if((tick&1)==0){FoundationWorld.WorldContextInput world=foundation.encodeWorldContext();Map<String,OnnxTensor> contextInputs=new HashMap<>();contextInputs.put("terrain",OnnxTensor.createTensor(environment,LongBuffer.wrap(world.terrain),new long[]{1,32,32}));contextInputs.put("city",OnnxTensor.createTensor(environment,LongBuffer.wrap(world.city),new long[]{1,32,32}));contextInputs.put("continuous",OnnxTensor.createTensor(environment,FloatBuffer.wrap(world.continuous),new long[]{1,7,32,32}));contextInputs.put("condition",OnnxTensor.createTensor(environment,FloatBuffer.wrap(world.condition),new long[]{1,15}));long contextBegan=System.nanoTime();try(OrtSession.Result result=contextSession.run(contextInputs)){context=((float[][])result.get(0).getValue())[0];}for(OnnxTensor tensor:contextInputs.values())tensor.close();milliseconds=(System.nanoTime()-contextBegan)/1_000_000.0;}
                        if(godMode)foundation.enforceSelectedGodMode();FoundationWorld.NeuralBatch batch = foundation.encodeNeural(); long groundedBegan = System.nanoTime();
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
                        if(tick%30==0){FoundationWorld.MacroInput macro=foundation.encodeMacro();Map<String,OnnxTensor> inputs=new HashMap<>();inputs.put("current",OnnxTensor.createTensor(environment,FloatBuffer.wrap(macro.current),new long[]{1,32,32,32}));inputs.put("previous",OnnxTensor.createTensor(environment,FloatBuffer.wrap(macro.previous),new long[]{1,32,32,32}));inputs.put("global_state",OnnxTensor.createTensor(environment,FloatBuffer.wrap(macro.global),new long[]{1,44}));inputs.put("previous_global",OnnxTensor.createTensor(environment,FloatBuffer.wrap(macro.previousGlobal),new long[]{1,44}));long began=System.nanoTime();try(OrtSession.Result result=macroSession.run(inputs)){foundation.applyMacro((float[][][][])result.get(0).getValue(),(float[][])result.get(1).getValue());}for(OnnxTensor tensor:inputs.values())tensor.close();macroMilliseconds=(System.nanoTime()-began)/1_000_000.0;}
                        if(tick%120==0){FoundationWorld.ColonyInput colony=foundation.encodeColony();Map<String,OnnxTensor> inputs=new HashMap<>();inputs.put("features",OnnxTensor.createTensor(environment,FloatBuffer.wrap(colony.features),new long[]{1,32,64}));inputs.put("mask",OnnxTensor.createTensor(environment,boolMatrix(colony.mask,1,32)));long began=System.nanoTime();try(OrtSession.Result result=colonySession.run(inputs)){foundation.applyColony((float[][][])result.get(0).getValue(),(float[][][])result.get(1).getValue());}for(OnnxTensor tensor:inputs.values())tensor.close();colonyMilliseconds=(System.nanoTime()-began)/1_000_000.0;}
                        if(tick%600==0){FoundationWorld.SocietyInput society=foundation.encodeSociety();try(OnnxTensor tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(society.features),new long[]{1,64})){Map<String,OnnxTensor> inputs=new HashMap<>();inputs.put("features",tensor);long began=System.nanoTime();try(OrtSession.Result result=societySession.run(inputs)){foundation.applySociety((float[][])result.get(0).getValue(),(float[][])result.get(1).getValue(),(float[][])result.get(2).getValue(),(float[][])result.get(3).getValue());}societyMilliseconds=(System.nanoTime()-began)/1_000_000.0;}}
                        if(tick%1500==0){float[] history=foundation.timelineFeatures();try(OnnxTensor tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(history),new long[]{1,24,64})){Map<String,OnnxTensor> inputs=new HashMap<>();inputs.put("sequence",tensor);long began=System.nanoTime();try(OrtSession.Result result=timelineSession.run(inputs)){foundation.applyTimeline((float[][])result.get(0).getValue(),(float[][])result.get(1).getValue(),(float[])result.get(2).getValue());}timelineMilliseconds=(System.nanoTime()-began)/1_000_000.0;}float[] batchHistory=new float[5*24*64];for(int i=0;i<5;i++)System.arraycopy(history,0,batchHistory,i*24*64,24*64);Map<String,OnnxTensor> inputs=new HashMap<>();inputs.put("sequence",OnnxTensor.createTensor(environment,FloatBuffer.wrap(batchHistory),new long[]{5,24,64}));inputs.put("action",OnnxTensor.createTensor(environment,LongBuffer.wrap(new long[]{0,1,2,3,4}),new long[]{5}));long began=System.nanoTime();try(OrtSession.Result result=counterfactualSession.run(inputs)){foundation.applyCounterfactual((float[])result.get(1).getValue(),(float[])result.get(2).getValue());}for(OnnxTensor tensor:inputs.values())tensor.close();counterfactualMilliseconds=(System.nanoTime()-began)/1_000_000.0;}
                        if(tick==0)Log.i(TAG,String.format(Locale.US,"COUPLED_ENSEMBLE_OK context=%.2f macro=%.2f colony=%.2f society=%.2f timeline=%.2f counterfactual=%.2f event=%d project=%d action=%d",milliseconds,macroMilliseconds,colonyMilliseconds,societyMilliseconds,timelineMilliseconds,counterfactualMilliseconds,foundation.timelineEvent,foundation.societyProject,foundation.counterfactualAction));
                        tickGrasper(environment,grasperSession);
                    }
                    advanceHabitat(1f / 30f);updatePerception(visibility,memory);control[0] = controlX; control[1] = controlY; control[2] = actionTouch ? 1f : 0f; control[3] = Math.max(-1f, Math.min(1f, cellularHealth * cellularNeural * 2f - 1f));
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
                        if ((tick & 7) == 0) {
                            long rasterBegan = System.nanoTime();
                            FoundationWorld.VaeInput liveVae=foundation==null?null:foundation.encodeSelectedVae();float[] vaeFeatures=liveVae==null?organismFeatures:liveVae.features,vaeMask=liveVae==null?organismMask:liveVae.mask;
                            try (OnnxTensor featureTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(vaeFeatures), new long[]{1, 576, 52});
                                 OnnxTensor maskTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(vaeMask), new long[]{1, 576})) {
                                Map<String, OnnxTensor> rasterInputs = new HashMap<>(); rasterInputs.put("features", featureTensor); rasterInputs.put("mask", maskTensor);
                                try (OrtSession.Result result = organismVaeSession.run(rasterInputs)) {float[][][][] rgba=(float[][][][])result.get(0).getValue();organismVaeFrame=rgbaBitmap(rgba);if(foundation!=null)foundation.applySelectedVae(rgba);}
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
            for(int index=0;index<foundation.creatures.size();index++){FoundationWorld.Creature creature=foundation.creatures.get(index);if(!creature.selected&&!isWorldVisible(creature.x,creature.y))continue;float sx=cx+creature.x-worldX,groundSy=cy+creature.y-worldY,sy=groundSy-creature.z;if(sx<-180||groundSy<-180||sx>width+180||groundSy>height+180)continue;float scale=creature.selected?4.4f:3.5f;
                float shadowScale=1f+creature.z/260f;paint.setColor(Color.argb(Math.max(28,100-(int)creature.z),0,0,0));canvas.drawOval(sx-28*scale/4*shadowScale,groundSy+16*scale,sx+28*scale/4*shadowScale,groundSy+22*scale,paint);
                paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(creature.selected?2.4f:1.2f);paint.setColor(creature.selected?Color.rgb(95,255,222):Color.argb(90,120,220,220));
                for(int edgeIndex=0;edgeIndex<creature.edges.length;edgeIndex++){if(!creature.edgeAlive[edgeIndex])continue;int[] edge=creature.edges[edgeIndex];float ax=sx+(creature.node[edge[0]][0]-creature.bodyProgress)*scale,ay=sy+creature.node[edge[0]][1]*scale,bx=sx+(creature.node[edge[1]][0]-creature.bodyProgress)*scale,by=sy+creature.node[edge[1]][1]*scale;canvas.drawLine(ax,ay,bx,by,paint);}paint.setStyle(Paint.Style.FILL);
                int fluidColor=creature.family==0?Color.rgb(204,45,76):creature.family==1?Color.rgb(123,73,167):creature.family==2?Color.rgb(115,211,66):creature.family==3?Color.rgb(195,77,255):Color.rgb(67,213,239);for(int p=0;p<FoundationWorld.CELL_PIXELS;p++){float surface=creature.cellState[9*FoundationWorld.CELL_PIXELS+p];if(surface>.035f){float puddleX=sx+((p%48)-24)*2.4f,puddleY=sy+22*scale+((p/48)-24)*1.25f;paint.setColor(Color.argb(Math.min(88,(int)(surface*130)),Color.red(fluidColor),Color.green(fluidColor),Color.blue(fluidColor)));canvas.drawCircle(puddleX,puddleY,2.2f+surface*5.5f,paint);}}
                for(FoundationWorld.Cell cell:creature.cells){if(cell.health<=.01f)continue;float px=cell.detached?cx+cell.detachedWorldX-worldX:sx+creature.cellX(cell)*scale,py=cell.detached?cy+cell.detachedWorldY-worldY:sy+creature.cellY(cell)*scale;int alpha=Math.max(35,Math.min(255,(int)(255*cell.alpha*cell.health))),red=Math.min(255,(int)(255*cell.red)),green=Math.min(255,(int)(255*cell.green)),blue=Math.min(255,(int)(255*cell.blue));paint.setColor(Color.argb(alpha,red,green,blue));float radius=(.72f+cell.sigma*.72f)*(creature.selected?2.05f:1.72f);canvas.drawRect(px-radius,py-radius,px+radius,py+radius,paint);}
                if(hudVisible&&labelsVisible){paint.setTextSize(12);paint.setColor(creature.selected?Color.rgb(95,255,222):Color.argb(190,190,220,220));String[] intents={"REST","FORAGE","HUNT","FLEE","MATE","FOLLOW","PHOTOSYNTHESIZE","MINE","PHASE FEED","REPAIR","GUARD","EXPLORE"};canvas.drawText(FoundationWorld.FAMILIES[creature.family]+(creature.selected?" // CONTROLLED":" // "+intents[Math.max(0,Math.min(intents.length-1,creature.intent))]),sx-52,sy+34*scale,paint);}
            }
        }
    }

    private void bar(Canvas canvas, float x, float y, float width, float value, int color, String label) {
        paint.setColor(Color.argb(180, 5, 11, 16)); canvas.drawRect(x, y, x + width, y + 18, paint); paint.setColor(color); canvas.drawRect(x + 2, y + 2, x + 2 + (width - 4) * Math.max(0f, Math.min(1f, value)), y + 16, paint); paint.setTextSize(13); paint.setColor(Color.WHITE); canvas.drawText(label, x + 6, y + 14, paint);
    }

    private void drawPerceptionOverlay(Canvas canvas,float cx,float cy){if(foundation==null||!hudVisible||!sightOverlay)return;FoundationWorld.Creature body=foundation.selected();float hearing=150f+body.sensory*150f,vision=430f+body.sensory*470f,angle=(float)Math.acos(.48),heading=(float)Math.atan2(aimY,aimX);Path cone=new Path();cone.moveTo(cx,cy);cone.lineTo(cx+(float)Math.cos(heading-angle)*vision,cy+(float)Math.sin(heading-angle)*vision);cone.lineTo(cx+(float)Math.cos(heading+angle)*vision,cy+(float)Math.sin(heading+angle)*vision);cone.close();paint.setColor(Color.argb(20,65,239,220));canvas.drawPath(cone,paint);paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(2);paint.setColor(Color.argb(105,65,239,220));canvas.drawPath(cone,paint);paint.setColor(Color.argb(120,170,120,255));canvas.drawCircle(cx,cy,hearing,paint);paint.setStyle(Paint.Style.FILL);}

    private void drawSetupSlider(Canvas canvas,String label,float value,float y,float width){float left=width*.34f,right=width*.76f;paint.setColor(Color.rgb(155,190,198));paint.setTextSize(16);canvas.drawText(label,left-width*.16f,y+6,paint);paint.setColor(Color.rgb(30,58,64));canvas.drawRoundRect(left,y-5,right,y+7,6,6,paint);paint.setColor(Color.rgb(67,239,220));canvas.drawRoundRect(left,y-5,left+(right-left)*value,y+7,6,6,paint);canvas.drawCircle(left+(right-left)*value,y+1,14,paint);paint.setColor(Color.WHITE);canvas.drawText(Math.round(value*100)+"",right+18,y+7,paint);}

    private void drawStartMenu(Canvas canvas,float width,float height){
        paint.setColor(Color.rgb(67,239,220));paint.setTextSize(Math.min(48,width*.035f));canvas.drawText("NULLVECTOR",width*.08f,height*.12f,paint);paint.setColor(Color.rgb(155,190,198));paint.setTextSize(17);canvas.drawText("COUPLED NEURAL CREATURE STAGE",width*.08f,height*.17f,paint);
        paint.setTextSize(14);canvas.drawText(status,width*.08f,height*.215f,paint);float cardLeft=width*.10f,cardWidth=width*.80f/5f,cardTop=height*.265f,cardBottom=height*.38f;
        for(int i=0;i<5;i++){float left=cardLeft+i*cardWidth;paint.setColor(i==setupFamily?Color.rgb(21,90,88):Color.rgb(8,24,30));canvas.drawRoundRect(left,cardTop,left+cardWidth-8,cardBottom,8,8,paint);paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(i==setupFamily?3:1);paint.setColor(i==setupFamily?Color.rgb(67,239,220):Color.rgb(66,105,112));canvas.drawRoundRect(left,cardTop,left+cardWidth-8,cardBottom,8,8,paint);paint.setStyle(Paint.Style.FILL);paint.setTextSize(14);paint.setColor(Color.WHITE);canvas.drawText(FoundationWorld.FAMILIES[i],left+10,(cardTop+cardBottom)*.5f+5,paint);}
        String[] modes={"SURVIVAL","GOD MODE"};for(int i=0;i<2;i++){float left=width*(.29f+i*.22f);paint.setColor(i==setupMode?Color.rgb(120,42,94):Color.rgb(20,24,31));canvas.drawRoundRect(left,height*.43f,left+width*.18f,height*.50f,8,8,paint);paint.setColor(Color.WHITE);paint.setTextSize(16);canvas.drawText(modes[i],left+20,height*.472f,paint);}
        drawSetupSlider(canvas,"LOCOMOTION",setupSpeed,height*.59f,width);drawSetupSlider(canvas,"SENSORY RANGE",setupSensory,height*.68f,width);drawSetupSlider(canvas,"RESILIENCE",setupResilience,height*.77f,width);
        paint.setColor(Color.rgb(67,239,220));canvas.drawRoundRect(width*.68f,height*.84f,width*.91f,height*.94f,10,10,paint);paint.setColor(Color.rgb(2,12,15));paint.setTextSize(20);canvas.drawText("ENTER LIVING WORLD",width*.71f,height*.902f,paint);paint.setColor(Color.rgb(150,180,188));paint.setTextSize(13);canvas.drawText("LEFT STICK MOVE · RIGHT STICK LOOK / AIM",width*.08f,height*.90f,paint);
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas); canvas.drawColor(Color.rgb(4, 9, 12)); float width = getWidth(), height = getHeight(); float cx = width * .5f, cy = height * .54f;
        if(!gameStarted){drawStartMenu(canvas,width,height);return;}
        int tile = 64, minX = (int)Math.floor((worldX - cx) / tile) - 1, maxX = (int)Math.ceil((worldX + cx) / tile) + 1;
        int minY = (int)Math.floor((worldY - cy) / tile) - 1, maxY = (int)Math.ceil((worldY + (height - cy)) / tile) + 1;
        if (neuralTerrain != null) { int chunk = 768, chunkMinX = (int)Math.floor((worldX - cx) / chunk) - 1, chunkMaxX = (int)Math.ceil((worldX + cx) / chunk) + 1, chunkMinY = (int)Math.floor((worldY - cy) / chunk) - 1, chunkMaxY = (int)Math.ceil((worldY + height - cy) / chunk) + 1; paint.setFilterBitmap(false); for (int py = chunkMinY; py <= chunkMaxY; py++) for (int px = chunkMinX; px <= chunkMaxX; px++) { float left = cx + px * chunk - worldX, top = cy + py * chunk - worldY; canvas.drawBitmap(neuralTerrain, null, new android.graphics.RectF(left, top, left + chunk, top + chunk), paint); } }
        for (int ty = minY; ty <= maxY; ty++) for (int tx = minX; tx <= maxX; tx++) {
            long hash = worldHash(tx, ty); int kind = (int)Math.floorMod(hash, 17); float sx = cx + tx * tile - worldX, sy = cy + ty * tile - worldY;
            if (neuralTerrain == null) { int base = 13 + (int)Math.floorMod(hash >>> 9, 8); if (kind < 3) paint.setColor(Color.rgb(8, 30 + base, 40 + base)); else if (kind < 6) paint.setColor(Color.rgb(27 + base, 25 + base, 20 + base / 2)); else paint.setColor(Color.rgb(8 + base / 2, 29 + base, 24 + base / 2)); canvas.drawRect(sx, sy, sx + tile + 1, sy + tile + 1, paint); }
            paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(1); paint.setColor(Color.argb(32, 95, 210, 190)); canvas.drawRect(sx, sy, sx + tile, sy + tile, paint); paint.setStyle(Paint.Style.FILL);
            if (Math.floorMod(hash >>> 22, 29) == 0) { paint.setColor(Color.rgb(155, 112, 64)); canvas.drawRect(sx + 23, sy + 20, sx + 41, sy + 46, paint); paint.setColor(Color.rgb(205, 159, 83)); canvas.drawCircle(sx + 32, sy + 19, 11, paint); }
            float tileWorldX=tx*tile+tile*.5f,tileWorldY=ty*tile+tile*.5f;if(!isWorldVisible(tileWorldX,tileWorldY)){paint.setColor(isWorldRemembered(tileWorldX,tileWorldY)?Color.argb(112,0,8,11):Color.argb(178,0,5,8));canvas.drawRect(sx,sy,sx+tile+1,sy+tile+1,paint);}
        }
        if(foundation!=null)synchronized(foundation){for(int p=0;p<FoundationWorld.MACRO_CELLS;p++)if(foundation.structures[p]>0){float wx=(p%32)*128+64,wy=(p/32)*128+64,sx=cx+wrappedDelta(worldX,wx),sy=cy+wrappedDelta(worldY,wy);if(sx<-90||sy<-90||sx>width+90||sy>height+90)continue;paint.setColor(Color.argb(220,20,31,39));canvas.drawRoundRect(sx-46,sy-46,sx+46,sy+46,9,9,paint);paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(4);int hue=foundation.structures[p];paint.setColor(hue==3?Color.rgb(255,95,128):hue==5?Color.rgb(177,104,255):Color.rgb(86,224,215));canvas.drawRoundRect(sx-46,sy-46,sx+46,sy+46,9,9,paint);paint.setStyle(Paint.Style.FILL);canvas.drawRect(sx-10,sy+25,sx+10,sy+48,paint);}}
        List<MaterialNode> materialSnapshot; synchronized(materials){materialSnapshot=new ArrayList<>(materials);}for (MaterialNode node : materialSnapshot) if (!node.isCreature()&&!node.isFragment()&&node.amount > 0f&&(isWorldVisible(node.x,node.y)||isWorldRemembered(node.x,node.y))) { float sx = cx + node.x - worldX, sy = cy + node.y - worldY; if (sx > -30 && sy > -30 && sx < width + 30 && sy < height + 30) { boolean visible=isWorldVisible(node.x,node.y);int color = node.type == 0 ? Color.rgb(151, 255, 68) : node.type == 1 ? Color.rgb(61, 206, 255) : Color.rgb(255, 190, 66); paint.setColor(Color.argb(visible?90:38, 0, 0, 0)); canvas.drawOval(sx - 13, sy + 7, sx + 13, sy + 14, paint); paint.setColor(visible?color:Color.rgb(63,88,86)); float radius = 5 + node.amount * 8; canvas.drawCircle(sx, sy, radius, paint); if(visible){paint.setColor(Color.argb(210, 235, 255, 235)); canvas.drawCircle(sx - radius * .28f, sy - radius * .28f, Math.max(2f, radius * .24f), paint);} } }
        List<Projectile> projectileSnapshot; synchronized(projectiles){projectileSnapshot=new ArrayList<>(projectiles);}for (Projectile shot : projectileSnapshot) {
            if(!isWorldVisible(shot.x,shot.y))continue;
            float sx = cx + shot.x - worldX, ground = cy + shot.y - worldY; float scale = 1f + shot.z / 260f;
            paint.setColor(Color.argb(90, 0, 0, 0)); canvas.drawOval(sx - 10 * scale, ground - 3, sx + 10 * scale, ground + 4, paint);
            int projectileColor=shot.ability==1?Color.rgb(255,177,59):shot.ability==2?Color.rgb(218,72,255):shot.ability==3?Color.rgb(65,239,220):(shot.resting?Color.rgb(160,130,76):Color.rgb(255,211,91));paint.setColor(projectileColor); canvas.drawCircle(sx, ground - shot.z, (shot.ability==2?11:7) * scale, paint); paint.setColor(Color.rgb(255, 245, 225)); canvas.drawCircle(sx - 2, ground - shot.z - 2, 2 * scale, paint);
        }
        drawPerceptionOverlay(canvas,cx,cy);drawFoundationCreatures(canvas,cx,cy,width,height);
        float organismSize = Math.min(250f, height * .31f); paint.setColor(Color.argb(115, 0, 0, 0));
        if(foundation==null) canvas.drawOval(cx - organismSize * .34f, cy + organismSize * .38f, cx + organismSize * .34f, cy + organismSize * .52f, paint);
        android.graphics.RectF organismRect = new android.graphics.RectF(cx - organismSize * .5f, cy - organismSize * .52f, cx + organismSize * .5f, cy + organismSize * .48f);
        paint.setFilterBitmap(false); paint.setAlpha(255);
        if (foundation==null && organismVaeFrame != null) canvas.drawBitmap(organismVaeFrame, null, organismRect, paint);
        if (foundation==null && cellularFrame != null) { paint.setAlpha(organismVaeFrame == null ? 255 : 76); canvas.drawBitmap(cellularFrame, null, organismRect, paint); paint.setAlpha(255); }
        if(hudVisible){paint.setColor(Color.argb(150, 255, 90, 180)); paint.setStrokeWidth(3); canvas.drawLine(cx, cy - organismSize * .12f, cx + aimX * 92f, cy - organismSize * .12f + aimY * 92f, paint); canvas.drawCircle(cx + aimX * 92f, cy - organismSize * .12f + aimY * 92f, 5, paint);
        paint.setColor(Color.rgb(67, 239, 220)); paint.setTextSize(25); canvas.drawText("NULLVECTOR // COUPLED NEURAL WORLD", 28, 39, paint); paint.setTextSize(15); paint.setColor(Color.rgb(165, 199, 199)); canvas.drawText("13-STAGE TEACHER ENSEMBLE · CELLULAR WORLD 4096² · MATERIAL " + gathered, 29, 63, paint);
        FoundationWorld.Creature selectedBody=foundation==null?null:foundation.selected();
        if(barsVisible){bar(canvas,28,78,158,cellularHealth,Color.rgb(62,224,115),"HEALTH");bar(canvas,28,102,158,cellularNeural,Color.rgb(214,72,255),"NEURAL");
        if(selectedBody!=null){bar(canvas,198,78,142,selectedBody.circulation,Color.rgb(245,77,101),"CIRCULATION");bar(canvas,198,102,142,selectedBody.respiration,Color.rgb(71,213,242),"RESPIRATION");bar(canvas,352,78,142,selectedBody.digestion,Color.rgb(245,174,62),"DIGESTION");bar(canvas,352,102,142,selectedBody.locomotion,Color.rgb(155,240,80),"LOCOMOTION");bar(canvas,506,78,142,selectedBody.sensory,Color.rgb(207,118,255),"SENSORY");bar(canvas,506,102,142,selectedBody.energy,Color.rgb(255,225,92),"ENERGY");}}
        if(foundation!=null){float cardWidth=Math.min(112,(width-40)/5);for(int i=0;i<5;i++){float left=20+i*cardWidth;paint.setColor(i==foundation.selected?Color.argb(220,16,75,75):Color.argb(190,5,17,22));canvas.drawRect(left,130,left+cardWidth-5,166,paint);paint.setColor(i==foundation.selected?Color.rgb(105,255,220):Color.rgb(160,190,200));paint.setTextSize(11);canvas.drawText(FoundationWorld.FAMILIES[i],left+5,152,paint);}}
        float moveCenterX=width*.13f,stickY=height*.84f,aimCenterX=width*.87f;paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(3);paint.setColor(movementTouch?Color.rgb(67,239,220):Color.argb(145,67,125,130));canvas.drawCircle(moveCenterX,stickY,72,paint);paint.setColor(actionTouch?Color.rgb(255,92,188):Color.argb(145,130,80,125));canvas.drawCircle(aimCenterX,stickY,72,paint);paint.setStyle(Paint.Style.FILL);paint.setColor(Color.rgb(67,239,220));canvas.drawCircle(moveCenterX+controlX*58,stickY+controlY*58,14,paint);paint.setColor(Color.rgb(255,92,188));canvas.drawCircle(aimCenterX+aimX*58,stickY+aimY*58,14,paint);paint.setTextSize(11);paint.setColor(Color.rgb(150,190,195));canvas.drawText("MOVE",moveCenterX-18,stickY+96,paint);canvas.drawText("LOOK / AIM",aimCenterX-34,stickY+96,paint);
        String[] actions={"GRASP","FEED","STRIKE","SCRAPE","CUT","THROW","FIRE"};float actionStart=width*.37f,actionGap=Math.min(84,width*.058f),actionY=height*.69f;for(int i=0;i<actions.length;i++){float x=actionStart+i*actionGap;paint.setColor(i==selectedAction?Color.rgb(132,42,103):Color.rgb(48,30,47));canvas.drawCircle(x,actionY,32,paint);paint.setColor(Color.WHITE);paint.setTextSize(10);canvas.drawText(actions[i],x-18,actionY+4,paint);}paint.setColor(Color.rgb(170,205,205));paint.setTextSize(12);canvas.drawText(actionStatus,actionStart,actionY+49,paint);
        paint.setColor(Color.argb(180, 5, 10, 14)); canvas.drawRect(width - 525, 18, width - 405, 58, paint); paint.setColor(Color.rgb(140, 205, 205)); paint.setTextSize(13); canvas.drawText(diagnostics ? "HIDE MODELS" : "MODEL INFO", width - 514, 43, paint);
        if (diagnostics) {
            float left = width * .47f, panelX = left + 18, panelY = 100; paint.setColor(Color.argb(238, 2, 6, 10)); canvas.drawRect(left, 72, width - 20, height * .73f, paint);
            paint.setColor(Color.rgb(67, 239, 220)); paint.setTextSize(17); canvas.drawText("NEURAL DEBUG // INTERNAL RUNTIME", panelX, panelY, paint);
            paint.setColor(Color.rgb(160, 190, 200)); paint.setTextSize(13); canvas.drawText(status, panelX, panelY + 22, paint);
            canvas.drawText(String.format("WORLD CONTEXT ENCODER · FP32 · STARTUP %.2f ms", milliseconds), panelX, panelY + 48, paint);
            canvas.drawText(String.format("RECURRENT ACTION CORE · %s · 30 Hz · %.2f ms", BuildConfig.SPLIT_ACTION ? "INT8" : "FP32", actionMilliseconds), panelX, panelY + 68, paint);
            canvas.drawText(String.format("CELL PHYSIOLOGY NCA · 492,492 PARAM · 15 Hz · %.2f ms", cellularMilliseconds), panelX, panelY + 88, paint);
            canvas.drawText(String.format("GROUNDED MUSCLE/CONTACT POLICY · 3.50M PARAM · 30 Hz · %.2f ms", groundedMilliseconds), panelX, panelY + 108, paint);
            canvas.drawText(String.format("ARTICULATED GRASPER POLICY · 8.01M PARAM · EVENT · %.2f ms", grasperMilliseconds), panelX, panelY + 128, paint);
            canvas.drawText(String.format("ECOLOGY INTENT/STEERING POLICY · 125,127 PARAM · 7.5 Hz · %.2f ms", ecologyMilliseconds), panelX, panelY + 148, paint);
            canvas.drawText(String.format("ORGANISM CELL VAE · 138,539 PARAM · 3.75 Hz · %.2f ms", organismVaeMilliseconds), panelX, panelY + 168, paint);
            canvas.drawText(String.format("WORLD FRAME VAE · 91,407 PARAM · DEBUG 2 Hz · %.2f ms", decoderMilliseconds), panelX, panelY + 188, paint);
            String ensemblePrecision=BuildConfig.SPLIT_ACTION?"INT8":"FP32";
            canvas.drawText(String.format("MACRO PATCH DYNAMICS · %s · 1 Hz · %.2f ms",ensemblePrecision,macroMilliseconds), panelX, panelY + 208, paint);
            canvas.drawText(String.format("COLONY ROLE POLICY · %s · 0.25 Hz · %.2f ms",ensemblePrecision,colonyMilliseconds), panelX, panelY + 228, paint);
            canvas.drawText(String.format("SOCIETY / BUILD POLICY · %s · 0.05 Hz · %.2f ms",ensemblePrecision,societyMilliseconds), panelX, panelY + 248, paint);
            canvas.drawText(String.format("TIMELINE + COUNTERFACTUAL · %s · %.2f + %.2f ms",ensemblePrecision,timelineMilliseconds,counterfactualMilliseconds), panelX, panelY + 268, paint);
            if(foundation!=null){String[] events={"QUIET","BIRTH","DEATH","PREDATION","MUTATION","COLONY","CLIMATE","CONSTRUCTION","DISCOVERY","MIGRATION"};String[] projects={"HABITAT","WORKSHOP","CLINIC","GRANARY","OBSERVATORY","GRAFT HOUSE","BATTERY HALL","SHRINE","MARKET"};canvas.drawText("FORECAST "+events[Math.max(0,Math.min(9,foundation.timelineEvent))]+" "+Math.round(foundation.timelineConfidence*100)+"% · PROJECT "+projects[Math.max(0,Math.min(8,foundation.societyProject))],panelX,panelY+288,paint);}
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
        paint.setTextSize(12);String[] toggleNames={sightOverlay?"SIGHT ON":"SIGHT OFF",labelsVisible?"LABELS ON":"LABELS OFF",barsVisible?"BARS ON":"BARS OFF"};for(int i=0;i<3;i++){float left=width-395+i*92;paint.setColor(Color.argb(190,5,16,20));canvas.drawRect(left,18,left+86,58,paint);paint.setColor(Color.rgb(135,210,202));canvas.drawText(toggleNames[i],left+8,43,paint);}
        }
        paint.setColor(Color.argb(220,5,16,20));canvas.drawRect(width-112,18,width-18,58,paint);paint.setColor(hudVisible?Color.rgb(140,205,205):Color.rgb(95,255,222));paint.setTextSize(13);canvas.drawText(hudVisible?"HUD OFF":"HUD ON",width-95,43,paint);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        float width=Math.max(1,getWidth()),height=Math.max(1,getHeight());int masked=event.getActionMasked(),actionIndex=event.getActionIndex();
        if(!gameStarted){if(masked==MotionEvent.ACTION_DOWN||masked==MotionEvent.ACTION_MOVE){float x=event.getX(actionIndex),y=event.getY(actionIndex);float cardLeft=width*.10f,cardWidth=width*.80f/5f;if(y>=height*.265f&&y<=height*.39f){int index=(int)((x-cardLeft)/cardWidth);if(index>=0&&index<5)setupFamily=index;}else if(y>=height*.42f&&y<=height*.51f){if(x>=width*.27f&&x<=width*.49f)setupMode=0;else if(x>=width*.49f&&x<=width*.73f)setupMode=1;}else if(y>=height*.55f&&y<=height*.81f){float value=Math.max(0,Math.min(1,(x-width*.34f)/(width*.42f)));if(y<height*.635f)setupSpeed=value;else if(y<height*.725f)setupSensory=value;else setupResilience=value;}else if(masked==MotionEvent.ACTION_DOWN&&x>=width*.66f&&x<=width*.93f&&y>=height*.82f&&y<=height*.96f&&foundation!=null){foundation.select(setupFamily);foundation.configureSelected(setupSpeed,setupSensory,setupResilience);godMode=setupMode==1;FoundationWorld.Creature body=foundation.selected();worldX=body.x;worldY=body.y;gameStarted=true;actionStatus=godMode?"GOD MODE · DAMAGE BYPASS ACTIVE":"SURVIVAL · PHYSIOLOGY ACTIVE";}}invalidate();return true;}
        if(event.getActionMasked()==MotionEvent.ACTION_DOWN&&event.getY()<75){float x=event.getX();if(x>width-125){hudVisible=!hudVisible;if(!hudVisible)diagnostics=false;invalidate();return true;}if(hudVisible&&x>width-217){barsVisible=!barsVisible;invalidate();return true;}if(hudVisible&&x>width-309){labelsVisible=!labelsVisible;invalidate();return true;}if(hudVisible&&x>width-401){sightOverlay=!sightOverlay;invalidate();return true;}if(hudVisible&&x>width-535&&x<width-395){diagnostics=!diagnostics;invalidate();return true;}}
        if(event.getActionMasked()==MotionEvent.ACTION_DOWN&&foundation!=null&&event.getY()>=126&&event.getY()<=174){int index=(int)((event.getX()-20)/Math.min(112,(width-40)/5));if(index>=0&&index<5){foundation.select(index);FoundationWorld.Creature selected=foundation.selected();worldX=selected.x;worldY=selected.y;invalidate();return true;}}
        if(masked==MotionEvent.ACTION_DOWN&&event.getY()>height*.62f&&event.getY()<height*.77f&&event.getX()>width*.32f&&event.getX()<width*.82f){float gap=Math.min(84,width*.058f),start=width*.37f;int index=Math.round((event.getX()-start)/gap);if(index>=0&&index<7){interactionAge=0;performAction(index);invalidate();return true;}}
        if(masked==MotionEvent.ACTION_DOWN||masked==MotionEvent.ACTION_POINTER_DOWN){int id=event.getPointerId(actionIndex);if(event.getX(actionIndex)<width*.5f&&movementPointer<0)movementPointer=id;else if(aimPointer<0)aimPointer=id;}
        if(masked==MotionEvent.ACTION_UP||masked==MotionEvent.ACTION_POINTER_UP||masked==MotionEvent.ACTION_CANCEL){int id=event.getPointerId(actionIndex);if(id==movementPointer)movementPointer=-1;if(id==aimPointer)aimPointer=-1;if(masked==MotionEvent.ACTION_CANCEL){movementPointer=aimPointer=-1;}}
        boolean move=false,act=false;for(int pointer=0;pointer<event.getPointerCount();pointer++){if((masked==MotionEvent.ACTION_POINTER_UP||masked==MotionEvent.ACTION_UP)&&pointer==actionIndex)continue;int id=event.getPointerId(pointer);float x=event.getX(pointer),y=event.getY(pointer);if(id==movementPointer){float dx=x-width*.13f,dy=y-height*.84f,length=Math.max(1,(float)Math.hypot(dx,dy)),scale=Math.min(1,length/72f);controlX=dx/length*scale;controlY=dy/length*scale;move=true;}if(id==aimPointer){float dx=x-width*.87f,dy=y-height*.84f,length=Math.max(1,(float)Math.hypot(dx,dy));if(length>8){aimX=dx/length;aimY=dy/length;}actionId=10;act=true;}}
        movementTouch=move;actionTouch=act;if(!move){controlX=controlY=0;}invalidate();return true;
    }

    @Override protected void onDetachedFromWindow() { running = false; super.onDetachedFromWindow(); }
}
