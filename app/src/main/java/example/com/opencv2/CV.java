package example.com.opencv2;

import android.os.Environment;
import android.util.Log;


import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;


import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Wajeeh on 7/21/2016.
 */
public class CV {

    public static void test() {
        FeatureDetector siftDetector = FeatureDetector.create(FeatureDetector.SIFT);
        DescriptorExtractor siftDescriptor =DescriptorExtractor.create(DescriptorExtractor.SIFT);
        DescriptorMatcher siftMatcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        List<Mat> trainingImages = new ArrayList<>();
        List<MatOfKeyPoint> trainingKeyPoints = new ArrayList<>();
        List<Mat> trainingDescriptors=new ArrayList<>();




        for (int i = 1; i < 50; i++) {
            Mat img = Highgui.imread(Environment.getExternalStorageDirectory().getAbsolutePath() + "/training_set/" + i + ".jpg");
            trainingImages.add(img);

        }

        siftDetector.detect(trainingImages,trainingKeyPoints);
        siftDescriptor.compute(trainingImages, trainingKeyPoints, trainingDescriptors);
        siftMatcher.add(trainingDescriptors);
        siftMatcher.train();
        Log.d("OPENCV ","TRAINED");





    }
}
