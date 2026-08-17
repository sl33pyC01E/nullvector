package world.nullvector.mobile;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Portable creature-stage scaffold. Neural policies choose contacts and muscle
 * activation; this class owns the causal skeleton, anchors, cells and world plane. */
final class FoundationWorld {
    static final int MAX_APPENDAGES = 8, MAX_MUSCLES = 60;
    static final String[] FAMILIES = {"HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE"};

    static final class Appendage {
        String kind; int side, segments, bend; float phase, rootX, rootY, endX, endY;
        boolean grounded() { return kind.equals("leg") || kind.equals("root") || kind.equals("wheel"); }
    }

    static final class Cell {
        float x, y; int tissue, appendage, component;
        float health=1f;
        float red,green,blue,alpha,sigma,offsetX,offsetY;
        final int[] nearest = new int[3]; final float[] weight = new float[3];
    }

    static final class Creature {
        final int family; final String genomeId; final float[] traits = new float[15];
        final Appendage[] appendages; final Cell[] cells; final float[][] rest, node, velocity;
        final int[][] edges; final int[] edgeAppendage, terminal; final float[] edgeLength;
        final float[][] muscles; final float[][] anchors; final boolean[] previousContact, contact;
        final float[] muscleActivation;
        float x, y, vx, vy, desiredX, desiredY, phase, bodyProgress, bodyVelocity, health = 1f, neural = 1f, energy = .82f;
        int intent = 11; float urgency = .5f;
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
            terminal=new int[appendages.length];java.util.Arrays.fill(terminal,-1);
            for(int i=0;i<edges.length;i++){JSONArray e=edgeArray.getJSONArray(i);edges[i][0]=e.getInt(0);edges[i][1]=e.getInt(1);edgeAppendage[i]=ownerArray.getInt(i);float dx=rest[edges[i][1]][0]-rest[edges[i][0]][0],dy=rest[edges[i][1]][1]-rest[edges[i][0]][1];edgeLength[i]=(float)Math.hypot(dx,dy);if(edgeAppendage[i]>=0)terminal[edgeAppendage[i]]=edges[i][1];}
            JSONArray muscleArray=skeleton.getJSONArray("muscles");muscles=new float[muscleArray.length()][];muscleActivation=new float[muscles.length];
            for(int i=0;i<muscles.length;i++){JSONArray m=muscleArray.getJSONArray(i);muscles[i]=new float[m.length()];for(int j=0;j<m.length();j++)muscles[i][j]=(float)m.getDouble(j);}
            JSONArray cellArray=row.getJSONArray("cells");cells=new Cell[cellArray.length()];
            for(int i=0;i<cells.length;i++){JSONObject value=cellArray.getJSONObject(i);JSONArray xy=value.getJSONArray("xy"),style=value.getJSONArray("neural_style");Cell c=new Cell();c.x=(float)xy.getDouble(0);c.y=(float)xy.getDouble(1);c.tissue=value.getInt("tissue");c.appendage=value.getInt("appendage");c.component=value.getInt("component");c.red=(float)style.getDouble(0);c.green=(float)style.getDouble(1);c.blue=(float)style.getDouble(2);c.alpha=(float)style.getDouble(3);c.sigma=(float)style.getDouble(4);c.offsetX=(float)style.getDouble(5);c.offsetY=(float)style.getDouble(6);cells[i]=c;skinWeights(c);}
            anchors=new float[appendages.length][2];previousContact=new boolean[appendages.length];contact=new boolean[appendages.length];for(float[] a:anchors){a[0]=Float.NaN;a[1]=Float.NaN;}
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

    final List<Creature> creatures = new ArrayList<>(); int selected = 0; float time;

    FoundationWorld(Context context) throws Exception {
        byte[] bytes; try(InputStream input=context.getAssets().open("foundation_anatomy.json")){bytes=input.readAllBytes();}
        JSONObject root=new JSONObject(new String(bytes, StandardCharsets.UTF_8));JSONArray rows=root.getJSONArray("organisms");
        for(int i=0;i<rows.length();i++)creatures.add(new Creature(rows.getJSONObject(i),i));creatures.get(0).selected=true;
    }

    synchronized Creature selected(){ return creatures.get(selected); }
    synchronized void select(int index){if(index<0||index>=creatures.size())return;creatures.get(selected).selected=false;selected=index;creatures.get(selected).selected=true;}

    synchronized void setPlayerControl(float x,float y){Creature c=selected();c.desiredX=x;c.desiredY=y;}

    synchronized NeuralBatch encodeNeural(){
        int count=creatures.size();NeuralBatch b=new NeuralBatch(count);
        for(int n=0;n<count;n++){Creature c=creatures.get(n);float speed=(float)Math.hypot(c.vx,c.vy);float sin=(float)Math.sin(Math.PI*2*c.phase),cos=(float)Math.cos(Math.PI*2*c.phase);
            b.global[n*23+c.family]=1;System.arraycopy(c.traits,0,b.global,n*23+5,15);b.global[n*23+20]=sin;b.global[n*23+21]=cos;b.global[n*23+22]=Math.min(1,speed/220f);
            for(int a=0;a<c.appendages.length;a++){int o=(n*MAX_APPENDAGES+a)*23;Appendage g=c.appendages[a];b.ownerMask[n*MAX_APPENDAGES+a]=true;int kind=kindIndex(g.kind);if(kind>=0)b.owner[o+kind]=1;b.owner[o+8]=g.side;b.owner[o+9]=g.segments/5f;b.owner[o+10]=(float)Math.sin(Math.PI*2*g.phase);b.owner[o+11]=(float)Math.cos(Math.PI*2*g.phase);b.owner[o+12]=g.rootX/24;b.owner[o+13]=g.rootY/24;b.owner[o+14]=g.endX/24;b.owner[o+15]=g.endY/24;int tip=c.terminal[a];b.owner[o+16]=(c.node[tip][0]-c.bodyProgress)/24;b.owner[o+17]=c.node[tip][1]/24;b.owner[o+18]=c.velocity[tip][0]/2;b.owner[o+19]=c.velocity[tip][1]/2;b.owner[o+20]=c.previousContact[a]?1:0;b.owner[o+21]=sin;b.owner[o+22]=cos;}
            for(int m=0;m<c.muscles.length;m++){float[] muscle=c.muscles[m];int o=(n*MAX_MUSCLES+m)*8,a=(int)muscle[2],joint=(int)muscle[6];Appendage g=c.appendages[a];b.muscleOwner[n*MAX_MUSCLES+m]=a;b.muscleMask[n*MAX_MUSCLES+m]=true;b.muscleMeta[o]=muscle[3];b.muscleMeta[o+1]=muscle[4];b.muscleMeta[o+2]=muscle[5];b.muscleMeta[o+3]=joint/5f;b.muscleMeta[o+4]=(float)Math.sin(Math.PI*2*g.phase);b.muscleMeta[o+5]=(float)Math.cos(Math.PI*2*g.phase);b.muscleMeta[o+6]=(float)Math.sin(Math.PI*2*joint/5f);b.muscleMeta[o+7]=(float)Math.cos(Math.PI*2*joint/5f);}
        }return b;
    }

    synchronized void applyNeural(float[][] muscles,float[][] logits,float[] drive){
        for(int n=0;n<creatures.size();n++){Creature c=creatures.get(n);float activity=Math.min(1f,(float)Math.hypot(c.desiredX,c.desiredY));
            for(int i=0;i<c.muscleActivation.length;i++){int owner=(int)c.muscles[i][2];c.muscleActivation[i]=muscles[n][i]*activity*appendageIntegrity(c,owner);}
            for(int a=0;a<c.appendages.length;a++)c.contact[a]=activity>.08f&&appendageIntegrity(c,a)>.22f&&c.appendages[a].grounded()&&logits[n][a]>=0;
        }
    }

    synchronized EcologyInput encodeEcology(int index){
        Creature c=creatures.get(index);EcologyInput input=new EcologyInput();input.self[c.family]=1;System.arraycopy(c.traits,0,input.self,5,15);
        for(int i=20;i<36;i++)input.self[i]=.45f;input.self[36+(c.family==0?0:c.family==1?1:c.family==2?2:c.family==3?4:5)]=1;
        input.self[46]=c.energy<.4f?1:0;input.self[47]=c.energy>=.4f?1:0;input.self[50]=c.health;input.self[51]=c.neural;for(int i=52;i<=56;i++)input.self[i]=Math.min(c.health,c.neural);input.self[57]=c.energy;input.self[58]=Math.min(1,c.energy*.7f);input.self[62]=c.vx/240;input.self[63]=c.vy/240;
        int[] appendageCounts=new int[8];for(Appendage a:c.appendages){int kind=kindIndex(a.kind);if(kind>=0)appendageCounts[kind]++;}for(int i=0;i<8;i++)input.self[67+i]=appendageCounts[i]/8f;
        for(int resource=0;resource<10;resource++){float scale=.0045f+resource*.00031f;float local=.5f+.5f*(float)Math.sin(c.x*scale+resource*1.7)* (float)Math.cos(c.y*scale*.83-resource*.61);float gx=(float)Math.cos(c.x*scale+resource*1.7)*scale*4,gy=-(float)Math.sin(c.y*scale*.83-resource*.61)*scale*.83f*4;int o=resource*4;input.resource[o]=local;input.resource[o+1]=gx;input.resource[o+2]=gy;input.resource[o+3]=dietAffinity(c.family,resource);}
        int cursor=0;for(int j=0;j<creatures.size()&&cursor<12;j++){if(j==index)continue;Creature other=creatures.get(j);float dx=shortDelta(c.x,other.x,4096),dy=shortDelta(c.y,other.y,4096),distance=Math.max(1,(float)Math.hypot(dx,dy)),nx=dx/distance,ny=dy/distance;int o=cursor*14;input.neighbor[o+other.family]=1;input.neighbor[o+5]=nx;input.neighbor[o+6]=ny;input.neighbor[o+7]=Math.min(1,distance/900);input.neighbor[o+8]=other.energy;input.neighbor[o+9]=other.energy*.7f;input.neighbor[o+10]=other.health;input.neighbor[o+11]=hostile(c.family,other.family)?1:0;input.neighbor[o+12]=c.family==other.family?1:0;input.neighbor[o+13]=1;input.mask[cursor]=1;cursor++;}
        input.intentIndex=index;return input;
    }

    synchronized void applyEcology(int index,float[] logits,float[] direction,float urgency){if(index==selected)return;Creature c=creatures.get(index);int intent=0;for(int i=1;i<logits.length;i++)if(logits[i]>logits[intent])intent=i;c.intent=intent;c.urgency=1f/(1f+(float)Math.exp(-urgency));float length=Math.max(.001f,(float)Math.hypot(direction[0],direction[1]));c.desiredX=direction[0]/length*c.urgency;c.desiredY=direction[1]/length*c.urgency;}

    synchronized void step(float dt){time+=dt;for(int i=0;i<creatures.size();i++)stepCreature(creatures.get(i),dt);}

    synchronized float[] selectedGrasperWorld(){Creature c=selected();int owner=-1;for(int i=0;i<c.appendages.length;i++){String kind=c.appendages[i].kind;if((kind.equals("arm")||kind.equals("tendril")||kind.equals("frond")||kind.equals("hardpoint"))&&appendageIntegrity(c,i)>.2f){owner=i;break;}}if(owner<0)return new float[]{c.x,c.y};int tip=c.terminal[owner];return new float[]{c.x+(c.node[tip][0]-c.bodyProgress)*4f,c.y+c.node[tip][1]*4f};}

    synchronized void feedSelected(float nutrition){Creature c=selected();c.energy=Math.min(1.2f,c.energy+Math.max(0,nutrition));c.health=Math.min(1,c.health+nutrition*.05f);}

    synchronized int attackSelected(float aimX,float aimY){Creature source=selected();float length=Math.max(.001f,(float)Math.hypot(aimX,aimY));aimX/=length;aimY/=length;Creature best=null;float bestAlong=Float.MAX_VALUE;for(Creature target:creatures){if(target==source||target.health<=0)continue;float dx=shortDelta(source.x,target.x,4096),dy=shortDelta(source.y,target.y,4096),along=dx*aimX+dy*aimY,side=Math.abs(dx*aimY-dy*aimX);if(along>0&&along<190&&side<52&&along<bestAlong){best=target;bestAlong=along;}}if(best==null)return 0;return damageCreature(best,source.x+aimX*bestAlong,source.y+aimY*bestAlong,42,.28f);}

    synchronized int impact(float worldX,float worldY,float radius,float damage){int affected=0;for(Creature target:creatures)affected+=damageCreature(target,worldX,worldY,radius,damage);return affected;}

    private int damageCreature(Creature target,float worldX,float worldY,float radius,float damage){int count=0;float total=0,neuralTotal=0,neuralCount=0;for(Cell cell:target.cells){if(cell.health<=0&&damage>=0)continue;float px=target.x+target.cellX(cell)*4f,py=target.y+target.cellY(cell)*4f,dx=shortDelta(worldX,px,4096),dy=shortDelta(worldY,py,4096),distance=(float)Math.hypot(dx,dy);if(distance<radius){cell.health=clamp(cell.health-damage*(1-distance/radius),0,1);count++;}total+=cell.health;if(cell.tissue==5){neuralTotal+=cell.health;neuralCount++;}}target.health=total/Math.max(1,target.cells.length);target.neural=neuralCount>0?neuralTotal/neuralCount:target.health;if(target.neural<.12f){target.desiredX=target.desiredY=0;}return count;}

    private static float appendageIntegrity(Creature c,int owner){float total=0,count=0;for(Cell cell:c.cells)if(cell.appendage==owner){total+=cell.health;count++;}return count>0?total/count:1;}

    private void stepCreature(Creature c,float dt){
        float activity=Math.min(1f,(float)Math.hypot(c.desiredX,c.desiredY));float cadence=.16f+activity*(.72f+.40f*c.traits[6]);c.phase=(c.phase+dt*cadence)%1f;
        float ground=Float.NEGATIVE_INFINITY;for(Appendage a:c.appendages)if(a.grounded())ground=Math.max(ground,a.endY);if(!Float.isFinite(ground))ground=maxRestY(c)+3;
        float[][] target=authoredPose(c,c.phase);
        for(int a=0;a<c.appendages.length;a++){if(c.contact[a]&&!c.previousContact[a]){int tip=c.terminal[a];c.anchors[a][0]=c.node[tip][0];c.anchors[a][1]=ground;}else if(!c.contact[a]){c.anchors[a][0]=Float.NaN;c.anchors[a][1]=Float.NaN;}}
        float contactDrive=0;int contacts=0;for(int a=0;a<c.appendages.length;a++)if(c.contact[a]){float desired=c.anchors[a][0]-target[c.terminal[a]][0];contactDrive+=clamp((desired-c.bodyProgress)*.18f,-.42f,.42f);contacts++;}if(contacts>0)contactDrive/=contacts;
        if(c.family==3)contactDrive+=.085f*activity;c.bodyVelocity=c.bodyVelocity*.86f+contactDrive*activity;c.bodyVelocity=clamp(c.bodyVelocity,-.55f,.55f);float priorProgress=c.bodyProgress;c.bodyProgress+=c.bodyVelocity;
        float[] fx=new float[c.node.length],fy=new float[c.node.length];for(int m=0;m<c.muscles.length;m++){float[] mm=c.muscles[m];int p=(int)mm[0],q=(int)mm[1];float dx=c.node[q][0]-c.node[p][0],dy=c.node[q][1]-c.node[p][1],len=Math.max(1e-5f,(float)Math.hypot(dx,dy));float strength=mm[3]*c.muscleActivation[m]*.085f*.72f,nx=-dy/len,ny=dx/len;fx[q]+=nx*strength;fy[q]+=ny*strength;fx[p]-=nx*strength*.35f;fy[p]-=ny*strength*.35f;}
        float rate=Math.min(1,dt*60);for(int n=0;n<c.node.length;n++){float tx=target[n][0]+c.bodyProgress,ty=target[n][1];float ax=(tx-c.node[n][0])*(n<componentCount(c)?.115f:.075f)+fx[n],ay=(ty-c.node[n][1])*(n<componentCount(c)?.115f:.075f)+fy[n];if(c.family!=3)ay+=.018f;c.velocity[n][0]=c.velocity[n][0]*.72f+ax*rate;c.velocity[n][1]=c.velocity[n][1]*.72f+ay*rate;c.node[n][0]+=c.velocity[n][0]*rate;c.node[n][1]+=c.velocity[n][1]*rate;}
        for(int iteration=0;iteration<6;iteration++){for(int e=0;e<c.edges.length;e++){int p=c.edges[e][0],q=c.edges[e][1];float dx=c.node[q][0]-c.node[p][0],dy=c.node[q][1]-c.node[p][1],len=Math.max(1e-5f,(float)Math.hypot(dx,dy)),cor=(len-c.edgeLength[e])/len*.5f;c.node[p][0]+=dx*cor;c.node[p][1]+=dy*cor;c.node[q][0]-=dx*cor;c.node[q][1]-=dy*cor;}c.node[0][0]+=(target[0][0]+c.bodyProgress-c.node[0][0])*.42f;c.node[0][1]+=(target[0][1]-c.node[0][1])*.42f;for(int a=0;a<c.appendages.length;a++)if(c.contact[a]){int tip=c.terminal[a];c.node[tip][0]=c.node[tip][0]*.08f+c.anchors[a][0]*.92f;c.node[tip][1]=c.node[tip][1]*.08f+c.anchors[a][1]*.92f;}}
        for(int n=0;n<c.node.length;n++){c.velocity[n][0]*=.38f;c.velocity[n][1]*=.38f;}System.arraycopy(c.contact,0,c.previousContact,0,c.contact.length);
        float gaitPixels=(c.bodyProgress-priorProgress)*7.2f;float desiredLen=Math.max(.001f,(float)Math.hypot(c.desiredX,c.desiredY));float desiredSpeed=activity*(120+105*c.traits[6]);float targetVx=c.desiredX/desiredLen*Math.max(Math.abs(gaitPixels)/Math.max(dt,.001f),desiredSpeed*.32f),targetVy=c.desiredY/desiredLen*Math.max(Math.abs(gaitPixels)/Math.max(dt,.001f),desiredSpeed*.32f);c.vx+=(targetVx-c.vx)*Math.min(1,dt*7);c.vy+=(targetVy-c.vy)*Math.min(1,dt*7);c.x=wrap(c.x+c.vx*dt,4096);c.y=wrap(c.y+c.vy*dt,4096);c.energy=Math.max(0,c.energy-dt*(.0003f+activity*.0012f));
    }

    private static float[][] authoredPose(Creature c,float phase){float[][] out=new float[c.rest.length][2];for(int i=0;i<out.length;i++){out[i][0]=c.rest[i][0];out[i][1]=c.rest[i][1];}float theta=(float)(Math.PI*2*phase);int components=componentCount(c);for(int i=0;i<components;i++){float leverage=clamp(Math.abs(c.rest[i][1])/15f,.18f,1);float sway=c.family==2?.28f:c.family==4?.12f:c.family==1?.16f:.65f;float bob=c.family==2?.34f:c.family==4?.10f:c.family==1?.42f:.72f;if(c.family==3){out[i][0]+=(float)Math.sin(theta*2+i*1.1)*(.22f+Math.abs(c.rest[i][1])/55);out[i][1]+=(float)Math.cos(theta*3+i*.8)*(.17f+Math.abs(c.rest[i][1])/70);}else{out[i][0]+=(float)Math.sin(theta+i*.37)*sway*leverage;out[i][1]-=Math.abs((float)Math.sin(theta+i*.21))*bob*(.55f+leverage*.45f);}}
        for(int a=0;a<c.appendages.length;a++){List<Integer> chain=new ArrayList<>();int rootComponent=-1;for(int e=0;e<c.edges.length;e++)if(c.edgeAppendage[e]==a){if(rootComponent<0){rootComponent=c.edges[e][0];chain.add(c.edges[e][1]);}else chain.add(c.edges[e][1]);}if(chain.size()<2)continue;Appendage g=c.appendages[a];float p=(phase+g.phase)%1f,stride=(new float[]{6.2f,7.2f,3.1f,5.2f,4.4f})[c.family],lift=(new float[]{3.8f,4.4f,2f,4.1f,2.8f})[c.family],stance=(new float[]{.60f,.50f,.74f,.54f,.68f})[c.family];float tx=g.endX,ty=g.endY;if(g.grounded()){if(p<stance)tx+=stride*(.5f-p/stance);else{float u=smooth((p-stance)/(1-stance));tx+=stride*(-.5f+u);ty-=(float)Math.sin(Math.PI*u)*lift;}}else{float gain=g.kind.equals("arm")?2.2f:g.kind.equals("tail")?3.2f:g.kind.equals("frond")?2.8f:g.kind.equals("tendril")?3.5f:.55f;tx+=(float)Math.sin(Math.PI*2*p)*gain;ty-=Math.abs((float)Math.sin(Math.PI*2*p))*gain*.34f;}float rootX=out[rootComponent][0]+g.rootX,rootY=out[rootComponent][1]+g.rootY;fabrik(out,chain,rootX,rootY,tx,ty,g.bend);}
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
}
