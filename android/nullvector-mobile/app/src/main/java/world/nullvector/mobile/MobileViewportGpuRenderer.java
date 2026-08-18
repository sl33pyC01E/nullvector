package world.nullvector.mobile;

import android.content.Context;
import android.graphics.Bitmap;

import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.gpu.GpuDelegate;
import com.google.android.gms.tflite.gpu.support.TfLiteGpu;
import com.google.android.gms.tflite.java.TfLite;

import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/** Recurrent action V5 followed by the trained full-viewport VAE on LiteRT GPU. */
final class MobileViewportGpuRenderer implements AutoCloseable {
    static final int LATENT_FLOATS = 48 * 32 * 32;
    static final int RGB_SIZE = 256;

    static final class Frame {
        final float[] latent;
        final Bitmap bitmap;
        final double packingMilliseconds, actionMilliseconds, decoderMilliseconds, totalMilliseconds;

        Frame(float[] latent, Bitmap bitmap, double packing, double action, double decoder, double total) {
            this.latent = latent; this.bitmap = bitmap; this.packingMilliseconds = packing;
            this.actionMilliseconds = action; this.decoderMilliseconds = decoder; this.totalMilliseconds = total;
        }
    }

    private final InterpreterApi action, decoder;
    private final GpuDelegate actionDelegate, decoderDelegate;
    private final Object[] actionInputs, decoderInputs;
    private final Map<String, ByteBuffer> actionBuffers = new HashMap<>();
    private final Map<Integer, Object> actionOutputs = new HashMap<>(), decoderOutputs = new HashMap<>();
    private final ByteBuffer actionLatent, decoderLatent, rgb;
    final String provider;

    private MobileViewportGpuRenderer(InterpreterApi action, InterpreterApi decoder, GpuDelegate actionDelegate, GpuDelegate decoderDelegate, String provider) {
        this.action = action; this.decoder = decoder; this.actionDelegate = actionDelegate; this.decoderDelegate = decoderDelegate; this.provider = provider;
        actionInputs = new Object[action.getInputTensorCount()]; decoderInputs = new Object[decoder.getInputTensorCount()];
        allocate("previous_latent", LATENT_FLOATS); allocate("spatial", 68 * 32 * 32); allocate("organism_field", 164 * 32 * 32);
        allocate("state", 64); allocate("actor_state", 128); allocate("actor_field", 8 * 32 * 32); allocate("visibility", 32 * 32);
        allocate("memory", 32 * 32); allocate("control", 4); allocate("action_one_hot", 22);
        actionLatent = directFloats(LATENT_FLOATS); actionOutputs.put(0, actionLatent);
        decoderLatent = directFloats(LATENT_FLOATS); decoderInputs[0] = decoderLatent;
        rgb = directFloats(3 * RGB_SIZE * RGB_SIZE); decoderOutputs.put(0, rgb);
    }

    static MobileViewportGpuRenderer create(Context context, String actionAsset, String decoderAsset) throws Exception {
        Tasks.await(TfLite.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build()));
        if (!Tasks.await(TfLiteGpu.isGpuDelegateAvailable(context))) {
            throw new IllegalStateException("LITERT GPU delegate unavailable on this device");
        }

        GpuDelegate actionDelegate = null, decoderDelegate = null;
        InterpreterApi action = null, decoder = null;
        try {
            actionDelegate = createGpuDelegate("viewport_action_v5_fp16.tflite");
            decoderDelegate = createGpuDelegate("frame_vae_mobile_v1_fp16.tflite");
            action = createInterpreter(context, actionAsset, actionDelegate, "viewport action");
            decoder = createInterpreter(context, decoderAsset, decoderDelegate, "viewport decoder");
            return new MobileViewportGpuRenderer(action, decoder, actionDelegate, decoderDelegate, "LITERT GPU FP16 WEIGHTS · ACTION V5 + MOBILE VAE");
        } catch (Throwable failure) {
            if (decoder != null) decoder.close();
            if (action != null) action.close();
            if (decoderDelegate != null) decoderDelegate.close();
            if (actionDelegate != null) actionDelegate.close();
            if (failure instanceof Exception) throw (Exception) failure;
            throw new IllegalStateException("Viewport LITERT GPU bootstrap failed: " + failure.getMessage(), failure);
        }
    }

    private static GpuDelegate createGpuDelegate(String modelTag) {
        try {
            return new GpuDelegate();
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("Unable to construct LiteRT GPU delegate for " + modelTag, failure);
        }
    }

    private static InterpreterApi createInterpreter(Context context, String modelAsset, GpuDelegate delegate, String stage) throws Exception {
        try {
            return InterpreterApi.create(readAsset(context, modelAsset), options(delegate));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("Unable to create LiteRT GPU interpreter for " + stage + " (" + modelAsset + "): " + failure.getMessage(), failure);
        }
    }

    private static InterpreterApi.Options options(GpuDelegate delegate) {
        InterpreterApi.Options options = new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
        if (delegate != null) options.addDelegate(delegate); else options.setUseXNNPACK(true).setNumThreads(8);
        return options;
    }

    private static ByteBuffer directFloats(int count) { return ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()); }

    private static ByteBuffer readAsset(Context context, String name) throws Exception {
        try (InputStream stream = context.getAssets().open(name); ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[64 * 1024]; for (int read; (read = stream.read(chunk)) >= 0;) bytes.write(chunk, 0, read);
            byte[] payload = bytes.toByteArray(); ByteBuffer result = ByteBuffer.allocateDirect(payload.length).order(ByteOrder.nativeOrder());
            result.put(payload).rewind(); return result;
        }
    }

    private int inputIndex(String suffix) {
        for (int index = 0; index < action.getInputTensorCount(); index++) if (action.getInputTensor(index).name().contains(suffix)) return index;
        throw new IllegalArgumentException("missing mobile viewport input " + suffix);
    }

    private void allocate(String name, int count) {
        ByteBuffer buffer = directFloats(count); actionBuffers.put(name, buffer); actionInputs[inputIndex(name)] = buffer;
    }

    private void write(String name, float[] values) {
        ByteBuffer buffer = actionBuffers.get(name); buffer.rewind(); buffer.asFloatBuffer().put(values); buffer.rewind();
    }

    private void writeNchwAsNhwc(String name, float[] values, int channels, int height, int width) {
        ByteBuffer buffer = actionBuffers.get(name); buffer.rewind(); FloatBuffer output = buffer.asFloatBuffer();
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) for (int channel = 0; channel < channels; channel++)
            output.put(values[(channel * height + y) * width + x]);
        buffer.rewind();
    }

    synchronized Frame run(float[] previousLatent, FoundationWorld.ViewportActionInput live, float[] visibility, float[] memory, float[] control, int actionId) {
        if (previousLatent == null || previousLatent.length != LATENT_FLOATS) throw new IllegalArgumentException("recurrent viewport latent is missing or malformed");
        long totalBegan = System.nanoTime(), stageBegan = totalBegan;
        writeNchwAsNhwc("previous_latent", previousLatent, 48, 32, 32); writeNchwAsNhwc("spatial", live.spatial, 68, 32, 32);
        writeNchwAsNhwc("organism_field", live.organismField, 164, 32, 32); write("state", live.state); write("actor_state", live.actorState);
        writeNchwAsNhwc("actor_field", live.actorField, 8, 32, 32); writeNchwAsNhwc("visibility", visibility, 1, 32, 32);
        writeNchwAsNhwc("memory", memory, 1, 32, 32); write("control", control);
        float[] oneHot = new float[22]; oneHot[Math.max(0, Math.min(21, actionId))] = 1; write("action_one_hot", oneHot);
        double packingMilliseconds = (System.nanoTime() - stageBegan) / 1_000_000.0;
        actionLatent.rewind(); stageBegan = System.nanoTime(); action.runForMultipleInputsOutputs(actionInputs, actionOutputs);
        double actionMilliseconds = (System.nanoTime() - stageBegan) / 1_000_000.0;
        actionLatent.rewind(); float[] nextLatent = new float[LATENT_FLOATS]; actionLatent.asFloatBuffer().get(nextLatent);
        decoderLatent.rewind(); FloatBuffer decoderInput = decoderLatent.asFloatBuffer();
        for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) for (int channel = 0; channel < 48; channel++)
            decoderInput.put(nextLatent[(channel * 32 + y) * 32 + x]);
        decoderLatent.rewind(); rgb.rewind(); stageBegan = System.nanoTime(); decoder.runForMultipleInputsOutputs(decoderInputs, decoderOutputs);
        double decoderMilliseconds = (System.nanoTime() - stageBegan) / 1_000_000.0;
        rgb.rewind(); FloatBuffer colors = rgb.asFloatBuffer(); int[] pixels = new int[RGB_SIZE * RGB_SIZE];
        for (int index = 0; index < pixels.length; index++) {
            int red = Math.max(0, Math.min(255, Math.round(colors.get() * 255))), green = Math.max(0, Math.min(255, Math.round(colors.get() * 255))), blue = Math.max(0, Math.min(255, Math.round(colors.get() * 255)));
            pixels[index] = 0xff000000 | (red << 16) | (green << 8) | blue;
        }
        Bitmap bitmap = Bitmap.createBitmap(pixels, RGB_SIZE, RGB_SIZE, Bitmap.Config.ARGB_8888);
        return new Frame(nextLatent, bitmap, packingMilliseconds, actionMilliseconds, decoderMilliseconds, (System.nanoTime() - totalBegan) / 1_000_000.0);
    }

    @Override public void close() { action.close(); decoder.close(); if (actionDelegate != null) actionDelegate.close(); if (decoderDelegate != null) decoderDelegate.close(); }
}
