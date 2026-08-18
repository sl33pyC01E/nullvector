package world.nullvector.mobile;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Portable creature-stage scaffold. Neural policies choose contacts and muscle
 * activation; this class owns the causal skeleton, anchors, cells and world plane. */
final class FoundationWorld {
    static final int MAX_APPENDAGES = 8, MAX_MUSCLES = 60;
    static final int MACRO_SIDE = 32, MACRO_CELLS = MACRO_SIDE * MACRO_SIDE, MACRO_CHANNELS = 32, GLOBAL_FEATURES = 44;
    static final int CELL_PIXELS = 48 * 48, CELL_STATIC_CHANNELS = 85, CELL_STATE_CHANNELS = 12, CELL_BOND_CHANNELS = 8;
    static final String[] FAMILIES = {"HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE"};

    static final class Appendage {
        String kind; int side, segments, bend; float phase, rootX, rootY, endX, endY;
        boolean grounded() { return kind.equals("leg") || kind.equals("root") || kind.equals("wheel"); }
    }

    static final class Cell {
        float x, y; int tissue, appendage, component, ncaX, ncaY;
        float health=1f;
        boolean detached; float detachedWorldX,detachedWorldY;
        float red,green,blue,alpha,sigma,offsetX,offsetY;
        final int[] nearest = new int[3]; final float[] weight = new float[3];
    }

    static final class Creature {
        final int family; final String genomeId; final float[] traits = new float[15];
        final Appendage[] appendages; final Cell[] cells; final float[][] rest, node, velocity;
        final int[][] edges; final int[] edgeAppendage, terminal; final float[] edgeLength; final boolean[] edgeAlive;
        final float[][] muscles; final float[][] anchors; final boolean[] previousContact, contact;
        final float[] muscleActivation;
        final boolean[] severed; final float[] severAge,severVx,severVy;
        final float[] cellStatic = new float[CELL_STATIC_CHANNELS * CELL_PIXELS];
        final float[] cellState = new float[CELL_STATE_CHANNELS * CELL_PIXELS];
        final float[] cellBonds = new float[CELL_BOND_CHANNELS * CELL_PIXELS];
        float x, y, z, vx, vy, vz, desiredX, desiredY, phase, bodyProgress, bodyVelocity, health = 1f, neural = 1f, energy = .82f, abilityCooldown; boolean carried;
        float circulation=1f, respiration=1f, digestion=1f, sensory=1f, locomotion=1f;
        int manipulationOwner=-1;boolean manipulationActive;float manipulationX,manipulationY,manipulationForce;
        int intent = 11; float urgency = .5f;
        int colonyRole; final float[] colonyAction = new float[3];
        boolean selected;

        Creature(JSONObject row, int ordinal) throws Exception {
            family = row.getInt("family_id"); genomeId = row.getString("genome_id");
            JSONArray traitArray = row.getJSONArray("traits"); for (int i=0;i<15;i++) traits[i]=(float)traitArray.getDouble(i);
            JSONArray appendageArray = row.getJSONArray("appendages"); appendages = new Appendage[appendageArray.length()];
            for (int i=0;i<appendages.length;i++) { JSONObject value=appendageArray.getJSONObject(i); Appendage a=new Appendage();
                a.kind=value.getString("kind");a.side=value.getInt("side");a.segments=value.getInt("segments");a.bend=value.getInt("bend");a.phase=(float)value.getDouble("phase");
                JSONArray root=value.getJSONArray("root_offset"),end=value.getJSONArray("endpoint");a.rootX=(float)root.getDouble(0);a.rootY=(float)root.getDouble(1);a.endX=(float)end.getDouble(0);a.endY=(float)end.getDouble(1);appendages[i]=a; }
            JSONObject skeleton=row.getJSONObject("skeleton"); JSONArray nodeArray=skeleton.getJSONArray("nodes"); rest=new float[nodeArray.length()][2];node=new float[nodeArray.length()][2];velocity=new float[nodeArray.length()][2];
            for(int i=0;i<rest.length;i++){JSONArray v=nodeArray.getJSONArray(i);rest[i][0]=node[i][0]=(float)v.getDouble(0);rest[i][1]=node[i][1]=(float)v.getDouble(1);}
            JSONArray edgeArray=skeleton.getJSONArray("edges"), ownerArray=skeleton.getJSONArray("edge_appendage");edges=new int[edgeArray.length()][2];edgeAppendage=new int[edges.length];edgeLength=new float[edges.length];
            terminal=new int[appendages.length];java.util.Arrays.fill(terminal,-1);edgeAlive=new boolean[edges.length];java.util.Arrays.fill(edgeAlive,true);
            for(int i=0;i<edges.length;i++){JSONArray e=edgeArray.getJSONArray(i);edges[i][0]=e.getInt(0);edges[i][1]=e.getInt(1);edgeAppendage[i]=ownerArray.getInt(i);float dx=rest[edges[i][1]][0]-rest[edges[i][0]][0],dy=rest[edges[i][1]][1]-rest[edges[i][0]][1];edgeLength[i]=(float)Math.hypot(dx,dy);if(edgeAppendage[i]>=0)terminal[edgeAppendage[i]]=edges[i][1];}
            JSONArray muscleArray=skeleton.getJSONArray("muscles");muscles=new float[muscleArray.length()][];muscleActivation=new float[muscles.length];
            for(int i=0;i<muscles.length;i++){JSONArray m=muscleArray.getJSONArray(i);muscles[i]=new float[m.length()];for(int j=0;j<m.length();j++)muscles[i][j]=(float)m.getDouble(j);}
            JSONArray cellArray=row.getJSONArray("cells");cells=new Cell[cellArray.length()];
            for(int i=0;i<cells.length;i++){JSONObject value=cellArray.getJSONObject(i);JSONArray xy=value.getJSONArray("xy"),nca=value.getJSONArray("nca_xy"),style=value.getJSONArray("neural_style");Cell c=new Cell();c.x=(float)xy.getDouble(0);c.y=(float)xy.getDouble(1);c.ncaX=nca.getInt(0);c.ncaY=nca.getInt(1);c.tissue=value.getInt("tissue");c.appendage=value.getInt("appendage");c.component=value.getInt("component");c.red=(float)style.getDouble(0);c.green=(float)style.getDouble(1);c.blue=(float)style.getDouble(2);c.alpha=(float)style.getDouble(3);c.sigma=(float)style.getDouble(4);c.offsetX=(float)style.getDouble(5);c.offsetY=(float)style.getDouble(6);cells[i]=c;skinWeights(c);}
            anchors=new float[appendages.length][2];previousContact=new boolean[appendages.length];contact=new boolean[appendages.length];severed=new boolean[appendages.length];severAge=new float[appendages.length];severVx=new float[appendages.length];severVy=new float[appendages.length];for(float[] a:anchors){a[0]=Float.NaN;a[1]=Float.NaN;}
            double angle=ordinal*Math.PI*2/5-Math.PI/2;x=2048f+(float)Math.cos(angle)*430f;y=2048f+(float)Math.sin(angle)*300f;phase=ordinal*.137f;
        }

        private void skinWeights(Cell cell){
            float[] best={Float.MAX_VALUE,Float.MAX_VALUE,Float.MAX_VALUE};
            for(int n=0;n<rest.length;n++){float d=(float)Math.hypot(cell.x-rest[n][0],cell.y-rest[n][1]);for(int k=0;k<3;k++)if(d<best[k]){for(int q=2;q>k;q--){best[q]=best[q-1];cell.nearest[q]=cell.nearest[q-1];}best[k]=d;cell.nearest[k]=n;break;}}
            float sum=0;for(int k=0;k<3;k++){cell.weight[k]=(float)Math.exp(-best[k]*.72);sum+=cell.weight[k];}for(int k=0;k<3;k++)cell.weight[k]/=Math.max(sum,1e-8f);
        }

        float cellX(Cell c){float value=c.x+c.offsetX;for(int k=0;k<3;k++){int n=c.nearest[k];value+=(node[n][0]-bodyProgress-rest[n][0])*c.weight[k];}return value;}
        float cellY(Cell c){float value=c.y+c.offsetY;for(int k=0;k<3;k++){int n=c.nearest[k];value+=(node[n][1]-rest[n][1])*c.weight[k];}return value;}
    }

    final List<Creature> creatures = new ArrayList<>(); int selected = 0; float time; boolean selectedAutonomous;
    final float[] resources = new float[10 * MACRO_CELLS];
    final int[] structures = new int[MACRO_CELLS];
    final float[] macroPrevious = new float[MACRO_CHANNELS * MACRO_CELLS];
    final float[] macroCurrent = new float[MACRO_CHANNELS * MACRO_CELLS];
    final float[] globalPrevious = new float[GLOBAL_FEATURES], globalCurrent = new float[GLOBAL_FEATURES];
    final float[] timelineHistory = new float[24 * 64]; int timelineRows;
    float settlementFood=.8f,settlementWealth=.5f,settlementPower=.35f,settlementKnowledge=.15f;
    int societyActivity,societyProject,societyDiplomacy,buildingCount;float timelineConfidence;int timelineEvent,counterfactualAction;
    final float[] societyLabor = new float[6];

    FoundationWorld(Context context) throws Exception {
        byte[] bytes; try(InputStream input=context.getAssets().open("foundation_anatomy.json")){bytes=input.readAllBytes();}
        JSONObject root=new JSONObject(new String(bytes, StandardCharsets.UTF_8));JSONArray rows=root.getJSONArray("organisms");
        for(int i=0;i<rows.length();i++)creatures.add(new Creature(rows.getJSONObject(i),i));
        float[] staticValues=floatAsset(context,"foundation_cell_static.f32",creatures.size()*CELL_STATIC_CHANNELS*CELL_PIXELS);
        float[] stateValues=floatAsset(context,"foundation_cell_state.f32",creatures.size()*CELL_STATE_CHANNELS*CELL_PIXELS);
        float[] bondValues=floatAsset(context,"foundation_cell_bonds.f32",creatures.size()*CELL_BOND_CHANNELS*CELL_PIXELS);
        for(int i=0;i<creatures.size();i++){Creature c=creatures.get(i);System.arraycopy(staticValues,i*c.cellStatic.length,c.cellStatic,0,c.cellStatic.length);System.arraycopy(stateValues,i*c.cellState.length,c.cellState,0,c.cellState.length);System.arraycopy(bondValues,i*c.cellBonds.length,c.cellBonds,0,c.cellBonds.length);}
        for(int channel=0;channel<10;channel++)for(int p=0;p<MACRO_CELLS;p++){int x=p%MACRO_SIDE,y=p/MACRO_SIDE;long hash=(x*0x9E3779B97F4A7C15L)^(y*0xC2B2AE3D27D4EB4FL)^(channel*0x165667B19E3779F9L);float wave=.5f+.25f*(float)Math.sin(x*.31+channel*.73)* (float)Math.cos(y*.27-channel*.41);resources[channel*MACRO_CELLS+p]=clamp(wave+((hash>>>57)&15)/120f,0,1);}
        MacroInput initial=encodeMacro();System.arraycopy(initial.current,0,macroCurrent,0,macroCurrent.length);System.arraycopy(initial.current,0,macroPrevious,0,macroPrevious.length);System.arraycopy(initial.global,0,globalCurrent,0,globalCurrent.length);System.arraycopy(initial.global,0,globalPrevious,0,globalPrevious.length);
        creatures.get(0).selected=true;
    }

    private static float[] floatAsset(Context context,String name,int count)throws Exception{byte[] bytes;try(InputStream input=context.getAssets().open(name)){bytes=input.readAllBytes();}FloatBuffer values=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();if(values.remaining()!=count)throw new IllegalStateException(name+" length");float[] result=new float[count];values.get(result);return result;}

    synchronized Creature selected(){ return creatures.get(selected); }
    synchronized void select(int index){if(index<0||index>=creatures.size())return;creatures.get(selected).selected=false;selected=index;creatures.get(selected).selected=true;}
    synchronized void setSelectedAutonomous(boolean value){selectedAutonomous=value;}

    synchronized void configureSelected(float speed,float sensory,float resilience){Creature c=selected();c.traits[6]=clamp(speed,0,1);c.traits[11]=clamp(resilience,0,1);c.sensory=.55f+.75f*clamp(sensory,0,1);c.energy=.72f+.35f*resilience;c.health=1;c.neural=1;c.circulation=c.respiration=c.digestion=c.locomotion=1;}

    synchronized void enforceSelectedGodMode(){Creature c=selected();c.health=c.neural=c.circulation=c.respiration=c.digestion=c.sensory=c.locomotion=1;c.energy=1.2f;for(Cell cell:c.cells)if(!cell.detached){cell.health=1;int p=cell.ncaY*48+cell.ncaX;c.cellState[p]=1;c.cellState[3*CELL_PIXELS+p]=1;c.cellState[4*CELL_PIXELS+p]=1;}}

    synchronized void carryCreature(Creature c,float x,float y){if(c==selected())return;c.carried=true;c.x=wrap(x,4096);c.y=wrap(y,4096);c.z=34;c.vx=c.vy=c.vz=0;c.desiredX=c.desiredY=0;}

    synchronized void throwCreature(Creature c,float aimX,float aimY,float strength){if(c==selected())return;float length=Math.max(.001f,(float)Math.hypot(aimX,aimY));c.carried=false;c.x=wrap(c.x,4096);c.y=wrap(c.y,4096);c.z=42;c.vx=aimX/length*strength;c.vy=aimY/length*strength;c.vz=245;c.desiredX=c.desiredY=0;}

    /** 1 machine kinetic, 2 anomaly phase, 3 acquired hybrid emitter. */
    synchronized int selectedProjectileAbility(){Creature c=selected();if(c.family==4)return 1;if(c.family==3)return 2;return c.traits[13]>.72f?3:0;}

    synchronized boolean consumeSelectedProjectileCost(){Creature c=selected();int ability=selectedProjectileAbility();float cost=ability==2?.115f:ability==1?.075f:.095f;if(ability==0||c.abilityCooldown>0||c.energy<cost)return false;c.energy-=cost;c.abilityCooldown=ability==1?.22f:ability==2?.48f:.34f;return true;}

    synchronized VaeInput encodeSelectedVae(){Creature c=selected();VaeInput result=new VaeInput();int row=0;for(Cell cell:c.cells){if(row>=576||cell.detached||cell.health<=.01f)continue;int offset=row*52;float localX=c.cellX(cell),localY=c.cellY(cell);result.features[offset]=clamp(localX/23.5f,-1,1);result.features[offset+1]=clamp(localY/23.5f,-1,1);int tissue=Math.max(0,Math.min(14,cell.tissue));result.features[offset+2+tissue]=1;result.features[offset+17+c.family]=1;for(int i=0;i<8;i++)result.features[offset+22+i]=c.traits[i];result.features[offset+30]=c.health;result.features[offset+31]=c.neural;result.features[offset+32]=c.energy;result.features[offset+33]=c.circulation;result.features[offset+34]=c.respiration;result.features[offset+35]=c.digestion;result.features[offset+36]=c.locomotion;int kind=0,side=0;if(cell.appendage>=0&&cell.appendage<c.appendages.length){Appendage appendage=c.appendages[cell.appendage];kind=Math.max(0,Math.min(8,kindIndex(appendage.kind)));side=appendage.side;}result.features[offset+37+kind]=1;result.features[offset+46]=side;result.features[offset+47]=(float)Math.sin(Math.PI*2*c.phase);result.features[offset+48]=(float)Math.cos(Math.PI*2*c.phase);result.features[offset+49]=cell.health;result.features[offset+50]=cell.appendage>=0?1:0;result.features[offset+51]=1;result.mask[row]=1;row++;}return result;}

    synchronized void applySelectedVae(float[][][][] rgba){Creature c=selected();for(Cell cell:c.cells){if(cell.detached||cell.health<=.01f)continue;int x=Math.max(0,Math.min(47,Math.round(c.cellX(cell)+23.5f))),y=Math.max(0,Math.min(47,Math.round(c.cellY(cell)+23.5f)));float predictedAlpha=clamp(rgba[0][3][y][x],0,1),luminance=(rgba[0][0][y][x]+rgba[0][1][y][x]+rgba[0][2][y][x])/3f;if(predictedAlpha<.2f||luminance<.04f)continue;float blend=.12f;cell.red+=(clamp(rgba[0][0][y][x],0,1)-cell.red)*blend;cell.green+=(clamp(rgba[0][1][y][x],0,1)-cell.green)*blend;cell.blue+=(clamp(rgba[0][2][y][x],0,1)-cell.blue)*blend;}}

    synchronized void setPlayerControl(float x,float y){if(selectedAutonomous)return;Creature c=selected();c.desiredX=x;c.desiredY=y;}

    synchronized NeuralBatch encodeNeural(){
        int count=creatures.size();NeuralBatch b=new NeuralBatch(count);
        for(int n=0;n<count;n++){Creature c=creatures.get(n);float speed=(float)Math.hypot(c.vx,c.vy);float sin=(float)Math.sin(Math.PI*2*c.phase),cos=(float)Math.cos(Math.PI*2*c.phase);
            b.global[n*23+c.family]=1;System.arraycopy(c.traits,0,b.global,n*23+5,15);b.global[n*23+20]=sin;b.global[n*23+21]=cos;b.global[n*23+22]=Math.min(1,speed/220f);
            for(int a=0;a<c.appendages.length;a++){int o=(n*MAX_APPENDAGES+a)*23;Appendage g=c.appendages[a];b.ownerMask[n*MAX_APPENDAGES+a]=true;int kind=kindIndex(g.kind);if(kind>=0)b.owner[o+kind]=1;b.owner[o+8]=g.side;b.owner[o+9]=g.segments/5f;b.owner[o+10]=(float)Math.sin(Math.PI*2*g.phase);b.owner[o+11]=(float)Math.cos(Math.PI*2*g.phase);b.owner[o+12]=g.rootX/24;b.owner[o+13]=g.rootY/24;b.owner[o+14]=g.endX/24;b.owner[o+15]=g.endY/24;int tip=c.terminal[a];b.owner[o+16]=(c.node[tip][0]-c.bodyProgress)/24;b.owner[o+17]=c.node[tip][1]/24;b.owner[o+18]=c.velocity[tip][0]/2;b.owner[o+19]=c.velocity[tip][1]/2;b.owner[o+20]=c.previousContact[a]?1:0;b.owner[o+21]=sin;b.owner[o+22]=cos;}
            for(int m=0;m<c.muscles.length;m++){float[] muscle=c.muscles[m];int o=(n*MAX_MUSCLES+m)*8,a=(int)muscle[2],joint=(int)muscle[6];Appendage g=c.appendages[a];b.muscleOwner[n*MAX_MUSCLES+m]=a;b.muscleMask[n*MAX_MUSCLES+m]=true;b.muscleMeta[o]=muscle[3];b.muscleMeta[o+1]=muscle[4];b.muscleMeta[o+2]=muscle[5];b.muscleMeta[o+3]=joint/5f;b.muscleMeta[o+4]=(float)Math.sin(Math.PI*2*g.phase);b.muscleMeta[o+5]=(float)Math.cos(Math.PI*2*g.phase);b.muscleMeta[o+6]=(float)Math.sin(Math.PI*2*joint/5f);b.muscleMeta[o+7]=(float)Math.cos(Math.PI*2*joint/5f);}
        }return b;
    }

    synchronized void applyNeural(float[][] muscles,float[][] logits,float[] drive){
        for(int n=0;n<creatures.size();n++){Creature c=creatures.get(n);float activity=Math.min(1f,(float)Math.hypot(c.desiredX,c.desiredY))*c.neural*c.locomotion;
            for(int i=0;i<c.muscleActivation.length;i++){int owner=(int)c.muscles[i][2];c.muscleActivation[i]=muscles[n][i]*activity*appendageIntegrity(c,owner);}
            for(int a=0;a<c.appendages.length;a++)c.contact[a]=activity>.08f&&appendageIntegrity(c,a)>.22f&&c.appendages[a].grounded()&&logits[n][a]>=0;
        }
    }

    synchronized EcologyInput encodeEcology(int index){
        Creature c=creatures.get(index);EcologyInput input=new EcologyInput();input.self[c.family]=1;System.arraycopy(c.traits,0,input.self,5,15);
        for(int i=20;i<36;i++)input.self[i]=.45f;input.self[36+(c.family==0?0:c.family==1?1:c.family==2?2:c.family==3?4:5)]=1;
        input.self[46]=c.energy<.4f?1:0;input.self[47]=c.energy>=.4f?1:0;input.self[50]=c.health;input.self[51]=c.neural;for(int i=52;i<=56;i++)input.self[i]=Math.min(c.health,c.neural);input.self[57]=c.energy;input.self[58]=Math.min(1,c.energy*.7f);input.self[62]=c.vx/240;input.self[63]=c.vy/240;
        int[] appendageCounts=new int[8];for(Appendage a:c.appendages){int kind=kindIndex(a.kind);if(kind>=0)appendageCounts[kind]++;}for(int i=0;i<8;i++)input.self[67+i]=appendageCounts[i]/8f;
        for(int resource=0;resource<10;resource++){float local=resourceAt(resource,c.x,c.y),gx=resourceAt(resource,c.x+128,c.y)-resourceAt(resource,c.x-128,c.y),gy=resourceAt(resource,c.x,c.y+128)-resourceAt(resource,c.x,c.y-128);int o=resource*4;input.resource[o]=local;input.resource[o+1]=gx;input.resource[o+2]=gy;input.resource[o+3]=dietAffinity(c.family,resource);}
        int cursor=0;for(int j=0;j<creatures.size()&&cursor<12;j++){if(j==index)continue;Creature other=creatures.get(j);float dx=shortDelta(c.x,other.x,4096),dy=shortDelta(c.y,other.y,4096),distance=Math.max(1,(float)Math.hypot(dx,dy)),nx=dx/distance,ny=dy/distance;int o=cursor*14;input.neighbor[o+other.family]=1;input.neighbor[o+5]=nx;input.neighbor[o+6]=ny;input.neighbor[o+7]=Math.min(1,distance/900);input.neighbor[o+8]=other.energy;input.neighbor[o+9]=other.energy*.7f;input.neighbor[o+10]=other.health;input.neighbor[o+11]=hostile(c.family,other.family)?1:0;input.neighbor[o+12]=c.family==other.family?1:0;input.neighbor[o+13]=1;input.mask[cursor]=1;cursor++;}
        input.intentIndex=index;return input;
    }

    synchronized void applyEcology(int index,float[] logits,float[] direction,float urgency){if(index==selected&&!selectedAutonomous)return;Creature c=creatures.get(index);int intent=0;for(int i=1;i<logits.length;i++)if(logits[i]>logits[intent])intent=i;c.intent=intent;c.urgency=1f/(1f+(float)Math.exp(-urgency));float length=Math.max(.001f,(float)Math.hypot(direction[0],direction[1]));c.desiredX=direction[0]/length*c.urgency;c.desiredY=direction[1]/length*c.urgency;}

    synchronized MacroInput encodeMacro(){
        java.util.Arrays.fill(macroCurrent,0);System.arraycopy(resources,0,macroCurrent,0,resources.length);
        float[] energy=new float[MACRO_CELLS],health=new float[MACRO_CELLS],count=new float[MACRO_CELLS];
        for(Creature c:creatures){int p=macroIndex(c.x,c.y);macroCurrent[(10+c.family)*MACRO_CELLS+p]+=1f/16f;energy[p]+=c.energy;health[p]+=c.health;count[p]++;macroCurrent[17*MACRO_CELLS+p]+=1f/8f;}
        for(int p=0;p<MACRO_CELLS;p++)if(count[p]>0){macroCurrent[15*MACRO_CELLS+p]=energy[p]/count[p];macroCurrent[16*MACRO_CELLS+p]=health[p]/count[p];}
        for(int p=0;p<MACRO_CELLS;p++)if(structures[p]>0){macroCurrent[(18+Math.floorMod(structures[p]-1,9))*MACRO_CELLS+p]=1;macroCurrent[28*MACRO_CELLS+p]=1;}
        float[] global=worldFeatures44();System.arraycopy(global,0,globalCurrent,0,GLOBAL_FEATURES);return new MacroInput(macroCurrent.clone(),macroPrevious.clone(),global,globalPrevious.clone());
    }

    synchronized void applyMacro(float[][][][] next,float[][] nextGlobal){
        System.arraycopy(macroCurrent,0,macroPrevious,0,macroCurrent.length);System.arraycopy(globalCurrent,0,globalPrevious,0,GLOBAL_FEATURES);
        for(int channel=0;channel<10;channel++)for(int y=0;y<MACRO_SIDE;y++)for(int x=0;x<MACRO_SIDE;x++){int p=y*MACRO_SIDE+x;float value=clamp(next[0][channel][y][x],0,1);resources[channel*MACRO_CELLS+p]=value;macroCurrent[channel*MACRO_CELLS+p]=value;}
        if(nextGlobal.length>0)for(int i=0;i<GLOBAL_FEATURES;i++)globalCurrent[i]=clamp(nextGlobal[0][i],0,1);
    }

    synchronized ColonyInput encodeColony(){float[] features=new float[32*64];boolean[] mask=new boolean[32];for(int i=0;i<creatures.size()&&i<32;i++){Creature c=creatures.get(i);mask[i]=true;int o=i*64;for(int j=0;j<15;j++)features[o+j]=c.traits[j];features[o+15+c.family]=1;features[o+20]=c.health;features[o+21]=c.neural;features[o+22]=c.circulation;features[o+23]=c.respiration;features[o+24]=c.digestion;features[o+25]=c.sensory;features[o+26]=c.locomotion;features[o+27]=c.energy;features[o+28]=Math.min(1,creatures.size()/16f);features[o+29]=settlementFood;features[o+30]=.65f;features[o+31]=(float)Math.sin(i*1.7);features[o+32]=(float)Math.cos(i*1.7);}return new ColonyInput(features,mask);}

    synchronized void applyColony(float[][][] roleLogits,float[][][] actions){for(int i=0;i<creatures.size();i++){Creature c=creatures.get(i);int role=0;for(int r=1;r<6;r++)if(roleLogits[0][i][r]>roleLogits[0][i][role])role=r;c.colonyRole=role;for(int a=0;a<3;a++)c.colonyAction[a]=actions[0][i][a];float local=resources[(c.family==4?2:c.family==3?4:8)*MACRO_CELLS+macroIndex(c.x,c.y)];if(role==0){float take=Math.min(local,.004f+.012f*c.colonyAction[0]);c.energy=Math.min(1.2f,c.energy+take);resources[(c.family==4?2:c.family==3?4:8)*MACRO_CELLS+macroIndex(c.x,c.y)]=Math.max(0,local-take);}if(role==3)c.health=Math.min(1,c.health+.004f*c.colonyAction[2]);}}

    synchronized SocietyInput encodeSociety(){float[] f=new float[64];for(int i=0;i<8;i++)f[i]=i<5?creatures.get(i%creatures.size()).traits[(i*2)%15]:.4f;f[18]=1;f[23]=Math.min(1,creatures.size()/24f);f[24]=Math.min(1,settlementWealth/4);f[25]=Math.min(1,settlementFood/3);f[26]=Math.min(1,settlementPower/3);float integrity=0;for(Creature c:creatures)integrity+=c.health;f[27]=integrity/creatures.size();f[30]=Math.min(1,settlementKnowledge);f[35]=Math.min(1,buildingCount/20f);f[38]=Math.min(1,time/5000);f[39]=1;f[47]=1;f[59]=buildingPurposeFraction(0);f[60]=buildingPurposeFraction(1);f[61]=buildingPurposeFraction(2);f[62]=buildingPurposeFraction(3);f[63]=Math.min(1,buildingCount/12f);return new SocietyInput(f);}

    synchronized void applySociety(float[][] activity,float[][] labor,float[][] diplomacy,float[][] project){societyActivity=argmax(activity[0]);societyDiplomacy=argmax(diplomacy[0]);societyProject=argmax(project[0]);float max=-Float.MAX_VALUE,sum=0;for(float v:labor[0])max=Math.max(max,v);for(int i=0;i<6;i++){societyLabor[i]=(float)Math.exp(labor[0][i]-max);sum+=societyLabor[i];}for(int i=0;i<6;i++)societyLabor[i]/=Math.max(sum,1e-6f);float harvest=.02f+.12f*societyLabor[0];settlementFood=Math.min(4,settlementFood+harvest);settlementWealth=Math.min(4,settlementWealth+.015f+.05f*societyLabor[5]);settlementKnowledge=Math.min(4,settlementKnowledge+.01f+.05f*societyLabor[4]);settlementPower=Math.min(4,settlementPower+.02f);if(settlementWealth>.62f&&settlementFood>.35f&&buildingCount<24){int center=macroIndex(creatures.get(0).x,creatures.get(0).y),cx=center%32,cy=center/32;int radius=2+buildingCount/5,angle=buildingCount*5;int x=Math.floorMod(cx+(int)Math.round(Math.cos(angle)*radius),32),y=Math.floorMod(cy+(int)Math.round(Math.sin(angle)*radius),32);for(int oy=-1;oy<=1;oy++)for(int ox=-1;ox<=1;ox++)if(Math.abs(ox)+Math.abs(oy)<=1)structures[Math.floorMod(y+oy,32)*32+Math.floorMod(x+ox,32)]=societyProject+1;buildingCount++;settlementWealth-=.55f;settlementFood-=.12f;}}

    synchronized float[] timelineFeatures(){float[] row=new float[64];float[] counts=new float[5];float health=0,energy=0;for(Creature c:creatures){counts[c.family]++;health+=c.health;energy+=c.energy;}row[0]=creatures.size()/180f;for(int i=0;i<5;i++)row[1+i]=counts[i]/creatures.size();row[7]=Math.min(1,buildingCount/20f);for(int r=0;r<10;r++){float total=0;for(int p=0;p<MACRO_CELLS;p++)total+=resources[r*MACRO_CELLS+p];row[12+r]=total/MACRO_CELLS;}row[30]=health/creatures.size();row[31]=energy/creatures.size();for(Creature c:creatures)row[40+Math.min(11,c.intent)]+=1f/creatures.size();System.arraycopy(timelineHistory,64, timelineHistory,0,23*64);System.arraycopy(row,0,timelineHistory,23*64,64);timelineRows=Math.min(24,timelineRows+1);if(timelineRows<24)for(int r=0;r<24-timelineRows;r++)System.arraycopy(row,0,timelineHistory,r*64,64);return timelineHistory.clone();}

    synchronized void applyTimeline(float[][] state,float[][] logits,float[] confidence){timelineEvent=argmax(logits[0]);timelineConfidence=confidence[0];}
    synchronized void applyCounterfactual(float[] benefit,float[] risk){counterfactualAction=0;float best=benefit[0]-.35f*risk[0];for(int i=1;i<benefit.length;i++){float score=benefit[i]-.35f*risk[i];if(score>best){best=score;counterfactualAction=i;}}}

    synchronized WorldContextInput encodeWorldContext(){long[] terrain=new long[MACRO_CELLS],city=new long[MACRO_CELLS];float[] continuous=new float[7*MACRO_CELLS],condition=new float[15];for(int p=0;p<MACRO_CELLS;p++){int best=0;for(int r=1;r<8;r++)if(resources[r*MACRO_CELLS+p]>resources[best*MACRO_CELLS+p])best=r;terrain[p]=best;city[p]=Math.min(7,Math.max(0,structures[p]));for(int r=0;r<7;r++)continuous[r*MACRO_CELLS+p]=resources[r*MACRO_CELLS+p];}condition[4]=1;for(Creature c:creatures)condition[6+c.family]+=1f/creatures.size();float season=time*.012f;condition[11]=(float)Math.sin(season);condition[12]=(float)Math.cos(season);condition[13]=Math.min(1,buildingCount/24f);condition[14]=Math.min(1,Math.abs(settlementFood-settlementPower)*.25f);return new WorldContextInput(terrain,city,continuous,condition);}

    /** Exact Android-side tensor contract for the trained whole-viewport action graph. */
    synchronized ViewportActionInput encodeViewportAction(float cameraX,float cameraY,float span){
        ViewportActionInput out=new ViewportActionInput();
        for(int gy=0;gy<32;gy++)for(int gx=0;gx<32;gx++){
            float wx=wrap(cameraX+((gx+.5f)/32f-.5f)*span,4096),wy=wrap(cameraY+((gy+.5f)/32f-.5f)*span,4096);int p=gy*32+gx,macro=macroIndex(wx,wy);
            for(int channel=0;channel<10;channel++)out.spatial[channel*1024+p]=resources[channel*1024+macro];
            out.spatial[10*1024+p]=1;int structure=structures[macro];if(structure>0){out.spatial[10*1024+p]=0;out.spatial[17*1024+p]=1;out.spatial[23*1024+p]=.92f;out.spatial[26*1024+p]=1;out.spatial[47*1024+p]=Math.min(1,structure/9f);}
            out.spatial[49*1024+p]=structure>0?1:0;out.spatial[57*1024+p]=1;out.spatial[63*1024+p]=structure>0?1:.1f;
        }
        float[] organismWeights=new float[1024];int row=0;for(Creature c:creatures){float dx=shortDelta(cameraX,c.x,4096),dy=shortDelta(cameraY,c.y,4096);if(Math.max(Math.abs(dx),Math.abs(dy))>span*.62f||row>=64)continue;int px=Math.max(0,Math.min(31,(int)Math.floor((dx/span+.5f)*32))),py=Math.max(0,Math.min(31,(int)Math.floor((dy/span+.5f)*32))),p=py*32+px;out.spatial[(27+c.family)*1024+p]=1;out.spatial[32*1024+p]=Math.max(out.spatial[32*1024+p],c.health);out.spatial[33*1024+p]=Math.max(out.spatial[33*1024+p],fluidRatio(c));out.spatial[34*1024+p]=Math.max(out.spatial[34*1024+p],scarMean(c));out.spatial[35*1024+p]=Math.max(out.spatial[35*1024+p],c.neural);out.spatial[36*1024+p]=Math.max(out.spatial[36*1024+p],Math.min(Math.min(c.circulation,c.respiration),c.digestion));out.spatial[37*1024+p]=c.health<=.01f?1:0;out.spatial[38*1024+p]=c==selected()?1:0;out.spatial[39*1024+p]=clamp(c.vx/240f,-1,1);out.spatial[40*1024+p]=clamp(c.vy/240f,-1,1);
            float[] actor=actorFeatures(c);int base=row*164;out.organisms[base]=dx/span;out.organisms[base+1]=dy/span;out.organisms[base+2]=c==selected()?1:0;out.organisms[base+3]=c.health>.01f?1:0;System.arraycopy(actor,0,out.organisms,base+4,128);for(int a=0;a<Math.min(8,c.appendages.length);a++){int tip=c.terminal[a],o=base+132+a*4;out.organisms[o]=(c.node[tip][0]-c.bodyProgress)/24f;out.organisms[o+1]=c.node[tip][1]/24f;out.organisms[o+2]=appendageIntegrity(c,a);out.organisms[o+3]=c.contact[a]?1:0;}for(int feature=0;feature<164;feature++)out.organismField[feature*1024+p]+=out.organisms[base+feature];organismWeights[p]++;out.organismMask[row]=true;row++;
        }
        for(int p=0;p<1024;p++)if(organismWeights[p]>1)for(int feature=0;feature<164;feature++)out.organismField[feature*1024+p]/=organismWeights[p];
        float[] actor=actorFeatures(selected());System.arraycopy(actor,0,out.actorState,0,128);fillActorField(selected(),out.actorField);fillWorldState(out.state);return out;
    }

    private float[] actorFeatures(Creature c){
        float[] row=new float[128];row[c.family]=1;row[7]=1;row[11+Math.max(0,Math.min(11,c.intent))]=1;row[23+c.family]=1;for(int i=0;i<15;i++)row[28+i]=c.traits[i];for(int i=0;i<16;i++)row[43+i]=c.traits[i%15];for(int r=0;r<10;r++)row[59+r]=dietAffinity(c.family,r);row[69]=c.health;row[70]=c.neural;row[71]=c.circulation;row[72]=c.respiration;row[73]=c.digestion;row[74]=c.sensory;row[75]=c.locomotion;row[76]=c.energy;row[77]=Math.min(1,c.energy*.7f);row[82]=c.energy;row[83]=c.health>.01f?1:0;row[84]=(c.neural<.12f||c.respiration<.08f)?1:0;row[85]=c.health<=.01f?1:0;row[86]=clamp(c.vx/240f,-1,1);row[87]=clamp(c.vy/240f,-1,1);float heading=(float)Math.atan2(c.vy,c.vx);row[88]=(float)Math.sin(heading);row[89]=(float)Math.cos(heading);row[90]=Math.min(1,c.cells.length/1024f);int alive=0,detached=0;float min=1,mean=0,variance=0;float[] tissueTotal=new float[15],tissueCount=new float[15];for(Cell cell:c.cells){if(cell.health>.01f)alive++;if(cell.detached)detached++;min=Math.min(min,cell.health);mean+=cell.health;int tissue=Math.max(0,Math.min(14,cell.tissue));tissueTotal[tissue]+=cell.health;tissueCount[tissue]++;}mean/=Math.max(1,c.cells.length);for(Cell cell:c.cells)variance+=(cell.health-mean)*(cell.health-mean);row[91]=alive/(float)Math.max(1,c.cells.length);row[92]=(alive-detached)/(float)Math.max(1,c.cells.length);row[93]=detached/(float)Math.max(1,c.cells.length);row[94]=Math.min(1,leakAmount(c)/8f);row[101]=contactRatio(c);row[102]=muscleMean(c);row[104]=muscleMax(c);row[105]=Math.min(1,c.appendages.length/32f);row[106]=Math.min(1,componentCount(c)/32f);for(int i=0;i<15;i++)row[109+i]=tissueCount[i]>0?tissueTotal[i]/tissueCount[i]:0;row[124]=fluidRatio(c);row[125]=scarMean(c);row[126]=min;row[127]=(float)Math.sqrt(variance/Math.max(1,c.cells.length));return row;
    }

    private void fillActorField(Creature c,float[] field){for(Cell cell:c.cells){if(cell.detached)continue;int x=Math.max(0,Math.min(31,Math.round(c.cellX(cell)*.6f+15.5f))),y=Math.max(0,Math.min(31,Math.round(c.cellY(cell)*.6f+15.5f))),p=y*32+x;field[p]=1;field[1024+p]=Math.max(field[1024+p],cell.health);field[2*1024+p]=Math.max(field[2*1024+p],c.cellState[CELL_PIXELS+cell.ncaY*48+cell.ncaX]);field[4*1024+p]=1;field[5*1024+p]=Math.max(field[5*1024+p],cell.tissue==5?1:0);field[6*1024+p]=Math.max(field[6*1024+p],cell.tissue>=5&&cell.tissue<=8?1:0);field[7*1024+p]=Math.max(field[7*1024+p],cell.appendage>=0?1:0);}}
    private void fillWorldState(float[] row){row[0]=creatures.size()/180f;for(Creature c:creatures)row[1+c.family]+=1f/creatures.size();row[7]=Math.min(1,buildingCount/20f);for(int r=0;r<10;r++){float total=0;for(int p=0;p<1024;p++)total+=resources[r*1024+p];row[12+r]=total/1024f;}row[22]=.72f;row[23]=.45f;row[24]=.4f;row[25]=.2f;row[28+Math.floorMod((int)(time*.003f),4)]=1;for(Creature c:creatures){row[32]+=c.health/creatures.size();row[33]+=c.neural/creatures.size();row[34]+=c.circulation/creatures.size();row[35]+=c.respiration/creatures.size();row[36]+=c.digestion/creatures.size();row[37]+=c.sensory/creatures.size();row[38]+=c.locomotion/creatures.size();row[39+Math.max(0,Math.min(11,c.intent))]+=1f/creatures.size();}}
    private static float fluidRatio(Creature c){float total=0,cap=0;for(int p=0;p<CELL_PIXELS;p++)if(c.cellStatic[p]>.5f){total+=c.cellState[CELL_PIXELS+p];cap++;}return total/Math.max(1,cap);}
    private static float scarMean(Creature c){float total=0,count=0;for(int p=0;p<CELL_PIXELS;p++)if(c.cellStatic[p]>.5f){total+=c.cellState[6*CELL_PIXELS+p];count++;}return total/Math.max(1,count);}
    private static float leakAmount(Creature c){float total=0;for(int p=0;p<CELL_PIXELS;p++)total+=c.cellState[9*CELL_PIXELS+p];return total/Math.max(1,c.cells.length);}
    private static float contactRatio(Creature c){float total=0;for(boolean value:c.contact)if(value)total++;return total/Math.max(1,c.contact.length);}
    private static float muscleMean(Creature c){float total=0;for(float value:c.muscleActivation)total+=value;return total/Math.max(1,c.muscleActivation.length);}
    private static float muscleMax(Creature c){float value=0;for(float item:c.muscleActivation)value=Math.max(value,item);return value;}

    synchronized boolean structureBlocked(float x,float y){return structures[macroIndex(x,y)]>0;}
    synchronized float resourceAt(int channel,float x,float y){return resources[Math.max(0,Math.min(9,channel))*MACRO_CELLS+macroIndex(x,y)];}
    private float[] worldFeatures44(){float[] row=new float[GLOBAL_FEATURES];row[0]=1;for(Creature c:creatures)row[18+c.family]+=1f/180f;row[27]=1;row[28]=1;row[29]=Math.min(1,buildingCount/24f);row[30]=Math.min(1,settlementWealth/8);row[31]=Math.min(1,settlementFood/8);row[32]=Math.min(1,settlementPower/8);row[33]=Math.min(1,buildingCount/24f);row[34]=Math.min(1,settlementKnowledge/4);row[35]=.7f;row[36]=1;return row;}
    private float buildingPurposeFraction(int purpose){int count=0;for(int value:structures)if(value==purpose+1)count++;return Math.min(1,count/24f);}
    private static int macroIndex(float x,float y){int gx=Math.floorMod((int)Math.floor(x/128f),32),gy=Math.floorMod((int)Math.floor(y/128f),32);return gy*32+gx;}
    private static int argmax(float[] values){int best=0;for(int i=1;i<values.length;i++)if(values[i]>values[best])best=i;return best;}

    synchronized void step(float dt){time+=dt;for(int i=0;i<creatures.size();i++)stepCreature(creatures.get(i),dt);resolveCreatureCollisions();}
    synchronized float[] positionSnapshot(){float[] values=new float[creatures.size()*2];for(int i=0;i<creatures.size();i++){values[i*2]=creatures.get(i).x;values[i*2+1]=creatures.get(i).y;}return values;}
    synchronized void rollbackPosition(int index,float x,float y){Creature c=creatures.get(index);c.x=x;c.y=y;c.vx*=-.08f;c.vy*=-.08f;}

    private void resolveCreatureCollisions(){for(int i=0;i<creatures.size();i++)for(int j=i+1;j<creatures.size();j++){Creature a=creatures.get(i),b=creatures.get(j);if(a.family==b.family||a.carried||b.carried||a.z>2||b.z>2)continue;float dx=shortDelta(a.x,b.x,4096),dy=shortDelta(a.y,b.y,4096),distance=Math.max(.001f,(float)Math.hypot(dx,dy)),radius=42+18*(a.traits[0]+b.traits[0]);if(distance>=radius)continue;float overlap=radius-distance,nx=dx/distance,ny=dy/distance;a.x=wrap(a.x-nx*overlap*.5f,4096);a.y=wrap(a.y-ny*overlap*.5f,4096);b.x=wrap(b.x+nx*overlap*.5f,4096);b.y=wrap(b.y+ny*overlap*.5f,4096);float closing=(a.vx-b.vx)*nx+(a.vy-b.vy)*ny;if(closing>0){a.vx-=nx*closing*.55f;a.vy-=ny*closing*.55f;b.vx+=nx*closing*.55f;b.vy+=ny*closing*.55f;if(hostile(a.family,b.family)&&closing>115){damageCreature(a,a.x+nx*radius*.4f,a.y+ny*radius*.4f,24,Math.min(.18f,closing/900));damageCreature(b,b.x-nx*radius*.4f,b.y-ny*radius*.4f,24,Math.min(.18f,closing/900));}}}}

    synchronized float[] selectedGrasperWorld(){Creature c=selected();int owner=-1;for(int i=0;i<c.appendages.length;i++){String kind=c.appendages[i].kind;if((kind.equals("arm")||kind.equals("tendril")||kind.equals("frond")||kind.equals("hardpoint"))&&appendageIntegrity(c,i)>.2f){owner=i;break;}}if(owner<0)return new float[]{c.x,c.y};int tip=c.terminal[owner];return new float[]{c.x+(c.node[tip][0]-c.bodyProgress)*4f,c.y+c.node[tip][1]*4f};}

    synchronized GrasperInput encodeGrasper(float targetX,float targetY,float mass,int goal,boolean attached){Creature c=selected();GrasperInput input=new GrasperInput();float dx=shortDelta(c.x,targetX,4096),dy=shortDelta(c.y,targetY,4096),distance=Math.max(.001f,(float)Math.hypot(dx,dy));String[] kinds={"arm","leg","tail","root","frond","tendril","wheel","hardpoint"};for(int a=0;a<c.appendages.length;a++){Appendage g=c.appendages[a];int o=a*16;for(int k=0;k<kinds.length;k++)if(kinds[k].equals(g.kind))input.owner[o+k]=1;input.owner[o+8]=g.side;input.owner[o+9]=g.segments/5f;input.owner[o+10]=(float)Math.sin(Math.PI*2*g.phase);input.owner[o+11]=(float)Math.cos(Math.PI*2*g.phase);input.owner[o+12]=g.rootX/24;input.owner[o+13]=g.rootY/24;input.owner[o+14]=g.endX/24;input.owner[o+15]=g.endY/24;input.mask[a]=!c.severed[a]&&appendageIntegrity(c,a)>.08f;}input.target[2]=1;input.target[4]=dx/distance;input.target[5]=dy/distance;input.target[6]=Math.min(1.25f,distance/96f);input.target[7]=Math.min(1,mass/4);input.target[8]=.72f;input.target[9]=1;input.target[10]=0;input.target[11]=goal==4?1:0;input.target[12+Math.max(0,Math.min(4,goal))]=1;input.target[17]=attached?1:0;input.global[c.family]=1;input.global[5]=(c.traits[3]+c.traits[4]+c.traits[5]+c.traits[6]+c.traits[7])/5;input.global[6]=(c.traits[8]+c.traits[9]+c.traits[10]+c.traits[11]+c.traits[12])/5;input.global[7]=(c.traits[13]+c.traits[14])/2;float fx=0,fy=0,count=0;for(Cell cell:c.cells){int p=cell.ncaY*48+cell.ncaX;float digestive=c.cellStatic[34*CELL_PIXELS+p]+c.cellStatic[35*CELL_PIXELS+p]+c.cellStatic[36*CELL_PIXELS+p];if(digestive>0){fx+=cell.x;fy+=cell.y;count++;}}input.global[8]=count>0?fx/count/24:0;input.global[9]=count>0?fy/count/24:0;return input;}

    synchronized GrasperCommand applyGrasper(float[] appendageLogits,float engage,float[] reach,float force,float release,float[] throwImpulse){Creature c=selected();int owner=0;for(int i=1;i<c.appendages.length;i++)if(appendageLogits[i]>appendageLogits[owner])owner=i;if(c.severed[owner]||appendageIntegrity(c,owner)<.08f){float best=-Float.MAX_VALUE;for(int i=0;i<c.appendages.length;i++)if(!c.severed[i]&&appendageIntegrity(c,i)>.08f&&appendageLogits[i]>best){owner=i;best=appendageLogits[i];}}c.manipulationOwner=owner;c.manipulationActive=engage>0&&!c.severed[owner];c.manipulationX=reach[0]*24;c.manipulationY=reach[1]*24;c.manipulationForce=force;return new GrasperCommand(owner,c.manipulationActive,release>0,throwImpulse[0],throwImpulse[1]);}

    synchronized float[] selectedGrasperWorld(int owner){Creature c=selected();if(owner<0||owner>=c.terminal.length)return new float[]{c.x,c.y};int tip=c.terminal[owner];return new float[]{c.x+(c.node[tip][0]-c.bodyProgress)*4f,c.y+c.node[tip][1]*4f};}
    synchronized float[] selectedFeederWorld(){Creature c=selected();float x=0,y=0,count=0;for(Cell cell:c.cells){int p=cell.ncaY*48+cell.ncaX;float digestive=c.cellStatic[34*CELL_PIXELS+p]+c.cellStatic[35*CELL_PIXELS+p]+c.cellStatic[36*CELL_PIXELS+p];if(digestive>0&&cell.health>.05f&&!cell.detached){x+=c.x+c.cellX(cell)*4;y+=c.y+c.cellY(cell)*4;count++;}}return count>0?new float[]{x/count,y/count}:new float[]{c.x,c.y};}

    synchronized void feedSelected(float nutrition){Creature c=selected();c.energy=Math.min(1.2f,c.energy+Math.max(0,nutrition));c.health=Math.min(1,c.health+nutrition*.05f);}

    synchronized void preparePhysiology(int index){Creature c=creatures.get(index);for(Cell cell:c.cells){int pixel=cell.ncaY*48+cell.ncaX;c.cellState[pixel]=Math.min(c.cellState[pixel],cell.health);if(cell.health<=.01f)c.cellState[11*CELL_PIXELS+pixel]=0;}}

    synchronized void addNutritionToSelected(float amount){Creature c=selected();for(Cell cell:c.cells){int p=cell.ncaY*48+cell.ncaX;if(c.cellStatic[(34*CELL_PIXELS)+p]+c.cellStatic[(35*CELL_PIXELS)+p]+c.cellStatic[(36*CELL_PIXELS)+p]>.1f){c.cellState[2*CELL_PIXELS+p]=clamp(c.cellState[2*CELL_PIXELS+p]+amount*.12f,0,1);c.cellState[3*CELL_PIXELS+p]=clamp(c.cellState[3*CELL_PIXELS+p]+amount*.07f,0,1);}}}

    synchronized void applyPhysiology(int index,float[] next){Creature c=creatures.get(index);System.arraycopy(next,0,c.cellState,0,c.cellState.length);for(Cell cell:c.cells){int p=cell.ncaY*48+cell.ncaX;float role=Math.min(1,c.cellStatic[37*CELL_PIXELS+p]+c.cellStatic[38*CELL_PIXELS+p]+c.cellStatic[39*CELL_PIXELS+p]);if(role>0){float health=c.cellState[p],oxygen=c.cellState[4*CELL_PIXELS+p],energy=c.cellState[3*CELL_PIXELS+p],weight=c.cellStatic[55*CELL_PIXELS+p];float homeostasis=role*weight*health*oxygen*(.3f+.7f*energy)*.78f;c.cellState[8*CELL_PIXELS+p]=Math.max(c.cellState[8*CELL_PIXELS+p],homeostasis);}cell.health=c.cellState[p];}c.health=bodyMean(c,0);c.energy=bodyMean(c,3);c.neural=organMean(c,8,37);c.circulation=organMean(c,1,28);c.respiration=organMean(c,4,31);c.digestion=organMean(c,3,34);c.sensory=systemHealth(c,40);c.locomotion=systemHealth(c,43);if(c.neural<.12f||c.respiration<.08f){c.desiredX=c.desiredY=0;} }

    private static float bodyMean(Creature c,int channel){float total=0,count=0;int offset=channel*CELL_PIXELS;for(int p=0;p<CELL_PIXELS;p++)if(c.cellStatic[p]>.5f){total+=c.cellState[offset+p];count++;}return count>0?total/count:0;}
    private static float organMean(Creature c,int channel,int staticStart){float total=0,count=0;int offset=channel*CELL_PIXELS;for(int p=0;p<CELL_PIXELS;p++){float mask=Math.min(1,c.cellStatic[staticStart*CELL_PIXELS+p]+c.cellStatic[(staticStart+1)*CELL_PIXELS+p]+c.cellStatic[(staticStart+2)*CELL_PIXELS+p]);total+=c.cellState[offset+p]*mask;count+=mask;}return count>0?total/count:c.health;}
    private static float systemHealth(Creature c,int staticStart){float total=0,count=0;for(int p=0;p<CELL_PIXELS;p++){float mask=Math.min(1,c.cellStatic[staticStart*CELL_PIXELS+p]+c.cellStatic[(staticStart+1)*CELL_PIXELS+p]+c.cellStatic[(staticStart+2)*CELL_PIXELS+p]);total+=c.cellState[p]*mask;count+=mask;}return count>0?total/count:c.health;}

    synchronized int attackSelected(float aimX,float aimY){return aimedDamage(aimX,aimY,42,.28f);}
    synchronized int scrapeSelected(float aimX,float aimY){return aimedDamage(aimX,aimY,28,.11f);}
    synchronized int cutSelected(float aimX,float aimY){return aimedDamage(aimX,aimY,18,.78f);}
    private int aimedDamage(float aimX,float aimY,float radius,float damage){Creature source=selected();float length=Math.max(.001f,(float)Math.hypot(aimX,aimY));aimX/=length;aimY/=length;Creature best=null;float bestAlong=Float.MAX_VALUE;for(Creature target:creatures){if(target==source||target.health<=0)continue;float dx=shortDelta(source.x,target.x,4096),dy=shortDelta(source.y,target.y,4096),along=dx*aimX+dy*aimY,side=Math.abs(dx*aimY-dy*aimX);if(along>0&&along<210&&side<58&&along<bestAlong){best=target;bestAlong=along;}}if(best==null)return 0;return damageCreature(best,source.x+aimX*bestAlong,source.y+aimY*bestAlong,radius,damage);}

    synchronized int impact(float worldX,float worldY,float radius,float damage){int affected=0;for(Creature target:creatures)affected+=damageCreature(target,worldX,worldY,radius,damage);return affected;}

    private int damageCreature(Creature target,float worldX,float worldY,float radius,float damage){int count=0;float total=0,neuralTotal=0,neuralCount=0;for(Cell cell:target.cells){if(cell.health<=0&&damage>=0)continue;float px=cell.detached?cell.detachedWorldX:target.x+target.cellX(cell)*4f,py=cell.detached?cell.detachedWorldY:target.y+target.cellY(cell)*4f,dx=shortDelta(worldX,px,4096),dy=shortDelta(worldY,py,4096),distance=(float)Math.hypot(dx,dy);if(distance<radius){cell.health=clamp(cell.health-damage*(1-distance/radius),0,1);if(cell.health<.34f)breakCellBonds(target,cell);count++;}total+=cell.health;if(cell.tissue==5){neuralTotal+=cell.health;neuralCount++;}}target.health=total/Math.max(1,target.cells.length);target.neural=neuralCount>0?neuralTotal/neuralCount:target.health;updateSevering(target);if(target.neural<.12f){target.desiredX=target.desiredY=0;}return count;}

    private static void breakCellBonds(Creature c,Cell cell){int[] dx={-1,0,1,-1,1,-1,0,1},dy={-1,-1,-1,0,0,1,1,1};int p=cell.ncaY*48+cell.ncaX;for(int d=0;d<8;d++){c.cellBonds[d*CELL_PIXELS+p]=0;int nx=cell.ncaX+dx[d],ny=cell.ncaY+dy[d];if(nx>=0&&ny>=0&&nx<48&&ny<48)c.cellBonds[(7-d)*CELL_PIXELS+ny*48+nx]=0;}}

    private static void updateSevering(Creature c){for(int owner=0;owner<c.appendages.length;owner++)if(!c.severed[owner]&&appendageIntegrity(c,owner)<.34f){c.severed[owner]=true;c.severVx[owner]=c.vx*.55f+(owner%2==0?-22f:22f);c.severVy[owner]=c.vy*.55f+18f;for(int e=0;e<c.edges.length;e++)if(c.edgeAppendage[e]==owner)c.edgeAlive[e]=false;for(Cell cell:c.cells)if(cell.appendage==owner){cell.detached=true;cell.detachedWorldX=wrap(c.x+c.cellX(cell)*4f,4096);cell.detachedWorldY=wrap(c.y+c.cellY(cell)*4f,4096);}}}

    private static float appendageIntegrity(Creature c,int owner){float total=0,count=0;for(Cell cell:c.cells)if(cell.appendage==owner){total+=cell.health;count++;}return count>0?total/count:1;}

    private void stepCreature(Creature c,float dt){
        c.abilityCooldown=Math.max(0,c.abilityCooldown-dt);if(c.carried)return;if(c.z>0){c.x=wrap(c.x+c.vx*dt,4096);c.y=wrap(c.y+c.vy*dt,4096);c.z+=c.vz*dt;c.vz-=560f*dt;if(c.z<=0){c.z=0;if(Math.abs(c.vz)>95){c.vz=-c.vz*.24f;c.vx*=.68f;c.vy*=.68f;}else{c.vz=0;c.vx*=.35f;c.vy*=.35f;}if(Math.hypot(c.vx,c.vy)>145)damageCreature(c,c.x,c.y,38,Math.min(.25f,(float)Math.hypot(c.vx,c.vy)/1200f));}return;}
        float activity=Math.min(1f,(float)Math.hypot(c.desiredX,c.desiredY));float roleSpeed=c.colonyRole==1?1.12f:c.colonyRole==2?1.04f:1f;float cadence=.16f+activity*(.72f+.40f*c.traits[6])*roleSpeed;c.phase=(c.phase+dt*cadence)%1f;
        float ground=Float.NEGATIVE_INFINITY;for(Appendage a:c.appendages)if(a.grounded())ground=Math.max(ground,a.endY);if(!Float.isFinite(ground))ground=maxRestY(c)+3;
        float[][] target=authoredPose(c,c.phase);
        for(int a=0;a<c.appendages.length;a++){if(c.contact[a]&&!c.previousContact[a]){int tip=c.terminal[a];c.anchors[a][0]=c.node[tip][0];c.anchors[a][1]=ground;}else if(!c.contact[a]){c.anchors[a][0]=Float.NaN;c.anchors[a][1]=Float.NaN;}}
        float contactDrive=0;int contacts=0;for(int a=0;a<c.appendages.length;a++)if(c.contact[a]){float desired=c.anchors[a][0]-target[c.terminal[a]][0];contactDrive+=clamp((desired-c.bodyProgress)*.18f,-.42f,.42f);contacts++;}if(contacts>0)contactDrive/=contacts;
        if(c.family==3)contactDrive+=.085f*activity;c.bodyVelocity=c.bodyVelocity*.86f+contactDrive*activity;c.bodyVelocity=clamp(c.bodyVelocity,-.55f,.55f);float priorProgress=c.bodyProgress;c.bodyProgress+=c.bodyVelocity;
        float[] fx=new float[c.node.length],fy=new float[c.node.length];for(int m=0;m<c.muscles.length;m++){float[] mm=c.muscles[m];int owner=(int)mm[2];if(owner>=0&&owner<c.severed.length&&c.severed[owner])continue;int p=(int)mm[0],q=(int)mm[1];float dx=c.node[q][0]-c.node[p][0],dy=c.node[q][1]-c.node[p][1],len=Math.max(1e-5f,(float)Math.hypot(dx,dy));float strength=mm[3]*c.muscleActivation[m]*.085f*.72f,nx=-dy/len,ny=dx/len;fx[q]+=nx*strength;fy[q]+=ny*strength;fx[p]-=nx*strength*.35f;fy[p]-=ny*strength*.35f;}
        float rate=Math.min(1,dt*60);for(int n=0;n<c.node.length;n++){float tx=target[n][0]+c.bodyProgress,ty=target[n][1];float ax=(tx-c.node[n][0])*(n<componentCount(c)?.115f:.075f)+fx[n],ay=(ty-c.node[n][1])*(n<componentCount(c)?.115f:.075f)+fy[n];if(c.family!=3)ay+=.018f;c.velocity[n][0]=c.velocity[n][0]*.72f+ax*rate;c.velocity[n][1]=c.velocity[n][1]*.72f+ay*rate;c.node[n][0]+=c.velocity[n][0]*rate;c.node[n][1]+=c.velocity[n][1]*rate;}
        for(int iteration=0;iteration<6;iteration++){for(int e=0;e<c.edges.length;e++){if(!c.edgeAlive[e])continue;int p=c.edges[e][0],q=c.edges[e][1];float dx=c.node[q][0]-c.node[p][0],dy=c.node[q][1]-c.node[p][1],len=Math.max(1e-5f,(float)Math.hypot(dx,dy)),cor=(len-c.edgeLength[e])/len*.5f;c.node[p][0]+=dx*cor;c.node[p][1]+=dy*cor;c.node[q][0]-=dx*cor;c.node[q][1]-=dy*cor;}c.node[0][0]+=(target[0][0]+c.bodyProgress-c.node[0][0])*.42f;c.node[0][1]+=(target[0][1]-c.node[0][1])*.42f;for(int a=0;a<c.appendages.length;a++)if(c.contact[a]&&!c.severed[a]){int tip=c.terminal[a];c.node[tip][0]=c.node[tip][0]*.08f+c.anchors[a][0]*.92f;c.node[tip][1]=c.node[tip][1]*.08f+c.anchors[a][1]*.92f;}}
        for(int n=0;n<c.node.length;n++){c.velocity[n][0]*=.38f;c.velocity[n][1]*=.38f;}System.arraycopy(c.contact,0,c.previousContact,0,c.contact.length);
        float gaitPixels=(c.bodyProgress-priorProgress)*7.2f;float desiredLen=Math.max(.001f,(float)Math.hypot(c.desiredX,c.desiredY));float capacity=c.neural*c.locomotion*Math.min(1,c.energy*1.4f);float desiredSpeed=activity*capacity*(120+105*c.traits[6]);float targetVx=c.desiredX/desiredLen*Math.max(Math.abs(gaitPixels)/Math.max(dt,.001f)*capacity,desiredSpeed*.32f),targetVy=c.desiredY/desiredLen*Math.max(Math.abs(gaitPixels)/Math.max(dt,.001f)*capacity,desiredSpeed*.32f);c.vx+=(targetVx-c.vx)*Math.min(1,dt*7);c.vy+=(targetVy-c.vy)*Math.min(1,dt*7);c.x=wrap(c.x+c.vx*dt,4096);c.y=wrap(c.y+c.vy*dt,4096);c.energy=Math.max(0,c.energy-dt*(.0003f+activity*.0012f));stepFragments(c,dt);
    }

    private static void stepFragments(Creature c,float dt){for(int owner=0;owner<c.severed.length;owner++)if(c.severed[owner]){c.severAge[owner]+=dt;float reconnect=(.8f+4.2f*c.traits[11])*(c.family==2?2.2f:c.family==4?1.8f:c.family==3?2.6f:1f);if(c.severAge[owner]<reconnect){float cx=0,cy=0,count=0;for(Cell cell:c.cells)if(cell.appendage==owner&&cell.detached){cx+=cell.detachedWorldX;cy+=cell.detachedWorldY;count++;}if(count>0){cx/=count;cy/=count;float dx=shortDelta(cx,c.x,4096),dy=shortDelta(cy,c.y,4096);c.severVx[owner]+=dx*dt*.55f;c.severVy[owner]+=dy*dt*.55f;}}c.severVx[owner]*=(float)Math.pow(.58,dt);c.severVy[owner]*=(float)Math.pow(.58,dt);for(Cell cell:c.cells)if(cell.appendage==owner&&cell.detached){cell.detachedWorldX=wrap(cell.detachedWorldX+c.severVx[owner]*dt,4096);cell.detachedWorldY=wrap(cell.detachedWorldY+c.severVy[owner]*dt,4096);if(c.severAge[owner]>reconnect&&(c.family==0||c.family==1))cell.health=Math.max(0,cell.health-dt*.018f);}}}

    private static float[][] authoredPose(Creature c,float phase){float[][] out=new float[c.rest.length][2];for(int i=0;i<out.length;i++){out[i][0]=c.rest[i][0];out[i][1]=c.rest[i][1];}float theta=(float)(Math.PI*2*phase);int components=componentCount(c);for(int i=0;i<components;i++){float leverage=clamp(Math.abs(c.rest[i][1])/15f,.18f,1);float sway=c.family==2?.28f:c.family==4?.12f:c.family==1?.16f:.65f;float bob=c.family==2?.34f:c.family==4?.10f:c.family==1?.42f:.72f;if(c.family==3){out[i][0]+=(float)Math.sin(theta*2+i*1.1)*(.22f+Math.abs(c.rest[i][1])/55);out[i][1]+=(float)Math.cos(theta*3+i*.8)*(.17f+Math.abs(c.rest[i][1])/70);}else{out[i][0]+=(float)Math.sin(theta+i*.37)*sway*leverage;out[i][1]-=Math.abs((float)Math.sin(theta+i*.21))*bob*(.55f+leverage*.45f);}}
        for(int a=0;a<c.appendages.length;a++){List<Integer> chain=new ArrayList<>();int rootComponent=-1;for(int e=0;e<c.edges.length;e++)if(c.edgeAppendage[e]==a){if(rootComponent<0){rootComponent=c.edges[e][0];chain.add(c.edges[e][1]);}else chain.add(c.edges[e][1]);}if(chain.size()<2)continue;Appendage g=c.appendages[a];float p=(phase+g.phase)%1f,stride=(new float[]{6.2f,7.2f,3.1f,5.2f,4.4f})[c.family],lift=(new float[]{3.8f,4.4f,2f,4.1f,2.8f})[c.family],stance=(new float[]{.60f,.50f,.74f,.54f,.68f})[c.family];float tx=g.endX,ty=g.endY;if(c.manipulationActive&&c.manipulationOwner==a){float blend=.18f+.72f*c.manipulationForce;tx+=(c.manipulationX-tx)*blend;ty+=(c.manipulationY-ty)*blend;}else if(g.grounded()){if(p<stance)tx+=stride*(.5f-p/stance);else{float u=smooth((p-stance)/(1-stance));tx+=stride*(-.5f+u);ty-=(float)Math.sin(Math.PI*u)*lift;}}else{float gain=g.kind.equals("arm")?2.2f:g.kind.equals("tail")?3.2f:g.kind.equals("frond")?2.8f:g.kind.equals("tendril")?3.5f:.55f;tx+=(float)Math.sin(Math.PI*2*p)*gain;ty-=Math.abs((float)Math.sin(Math.PI*2*p))*gain*.34f;}float rootX=out[rootComponent][0]+g.rootX,rootY=out[rootComponent][1]+g.rootY;fabrik(out,chain,rootX,rootY,tx,ty,g.bend);}
        return out;}

    private static void fabrik(float[][] nodes,List<Integer> chain,float rootX,float rootY,float targetX,float targetY,int bend){int size=chain.size();float[] length=new float[size-1];for(int i=0;i<size-1;i++){int a=chain.get(i),b=chain.get(i+1);length[i]=(float)Math.hypot(nodes[b][0]-nodes[a][0],nodes[b][1]-nodes[a][1]);}int first=chain.get(0);nodes[first][0]=rootX;nodes[first][1]=rootY;for(int iteration=0;iteration<7;iteration++){int last=chain.get(size-1);nodes[last][0]=targetX;nodes[last][1]=targetY;for(int i=size-2;i>=0;i--)placeAtDistance(nodes,chain.get(i),chain.get(i+1),length[i]);nodes[first][0]=rootX;nodes[first][1]=rootY;for(int i=1;i<size;i++)placeAtDistance(nodes,chain.get(i),chain.get(i-1),length[i-1]);}}
    private static void placeAtDistance(float[][] n,int moving,int fixed,float distance){float dx=n[moving][0]-n[fixed][0],dy=n[moving][1]-n[fixed][1],len=Math.max(1e-5f,(float)Math.hypot(dx,dy));n[moving][0]=n[fixed][0]+dx/len*distance;n[moving][1]=n[fixed][1]+dy/len*distance;}
    private static int componentCount(Creature c){int max=-1;for(Cell cell:c.cells)max=Math.max(max,cell.component);return max+1;}
    private static float maxRestY(Creature c){float value=-Float.MAX_VALUE;for(float[] n:c.rest)value=Math.max(value,n[1]);return value;}
    private static float smooth(float v){return v*v*(3-2*v);}private static float clamp(float v,float a,float b){return Math.max(a,Math.min(b,v));}private static float wrap(float v,float size){v%=size;return v<0?v+size:v;}
    private static int kindIndex(String value){String[] kinds={"arm","leg","tail","root","frond","tendril","wheel","hardpoint"};for(int i=0;i<kinds.length;i++)if(kinds[i].equals(value))return i;return -1;}
    private static float dietAffinity(int family,int resource){if(family==2)return resource==1||resource==0?.95f:.12f;if(family==3)return resource==4||resource==6?.9f:.1f;if(family==4)return resource==2||resource==3?.95f:.08f;if(family==1)return resource==8||resource==9?.95f:.12f;return resource==8||resource==9||resource==2?.68f:.18f;}
    private static boolean hostile(int left,int right){return left!=right&&(left==0||left==1||left==4||right==3);}private static float shortDelta(float a,float b,float size){float d=b-a;if(d>size*.5f)d-=size;if(d<-size*.5f)d+=size;return d;}

    static final class NeuralBatch {final int count;final float[] owner,global,muscleMeta;final boolean[] ownerMask,muscleMask;final long[] muscleOwner;NeuralBatch(int n){count=n;owner=new float[n*MAX_APPENDAGES*23];global=new float[n*23];ownerMask=new boolean[n*MAX_APPENDAGES];muscleMeta=new float[n*MAX_MUSCLES*8];muscleOwner=new long[n*MAX_MUSCLES];muscleMask=new boolean[n*MAX_MUSCLES];}}
    static final class EcologyInput {final float[] self=new float[94],resource=new float[40],neighbor=new float[168],mask=new float[12];int intentIndex;}
    static final class GrasperInput {final float[] owner=new float[MAX_APPENDAGES*16],target=new float[18],global=new float[10];final boolean[] mask=new boolean[MAX_APPENDAGES];}
    static final class GrasperCommand {final int owner;final boolean engage,release;final float throwX,throwY;GrasperCommand(int owner,boolean engage,boolean release,float throwX,float throwY){this.owner=owner;this.engage=engage;this.release=release;this.throwX=throwX;this.throwY=throwY;}}
    static final class MacroInput {final float[] current,previous,global,previousGlobal;MacroInput(float[] c,float[] p,float[] g,float[] pg){current=c;previous=p;global=g;previousGlobal=pg;}}
    static final class ColonyInput {final float[] features;final boolean[] mask;ColonyInput(float[] f,boolean[] m){features=f;mask=m;}}
    static final class SocietyInput {final float[] features;SocietyInput(float[] f){features=f;}}
    static final class WorldContextInput {final long[] terrain,city;final float[] continuous,condition;WorldContextInput(long[] t,long[] c,float[] v,float[] k){terrain=t;city=c;continuous=v;condition=k;}}
    static final class VaeInput {final float[] features=new float[576*52],mask=new float[576];}
    static final class ViewportActionInput {final float[] spatial=new float[68*32*32],organisms=new float[64*164],organismField=new float[164*32*32],state=new float[64],actorState=new float[128],actorField=new float[8*32*32];final boolean[] organismMask=new boolean[64];}
}
