package world.nullvector.mobile;

import android.app.Activity;
import android.os.Bundle;

public final class MainActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        // The theme owns fullscreen layout. Calling WindowInsetsController.hide
        // here races Android 16's startup animation thread and can crash inside
        // InsetsAnimationControlImpl before our view receives its first frame.
        setContentView(new NeuralWorldView(this));
    }
}
