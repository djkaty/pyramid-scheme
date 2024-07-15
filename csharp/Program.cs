using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenCvSharp;

public class ImageStacking
{
    private static double ComputeFv(Mat image, int gaussianBlurKernelSize = 5, int laplacianKernelSize = 5)
    {
        Mat gray = new Mat();
        Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

        Mat blurred = new Mat();
        Cv2.GaussianBlur(gray, blurred, new Size(gaussianBlurKernelSize, gaussianBlurKernelSize), 0);

        Mat laplacian = new Mat();
        Cv2.Laplacian(blurred, laplacian, MatType.CV_64F, ksize: laplacianKernelSize);

        Mat mean = new Mat(), stddev = new Mat();
        Cv2.MeanStdDev(laplacian, mean, stddev);

        double stdDevValue = stddev.At<double>(0, 0) / 25.0;
        return stdDevValue * stdDevValue;
    }

    private static List<Mat> StackFilter(List<Mat> images, double threshold)
    {
        List<double> focus = images.Select(ComputeFv).ToList();
        Console.WriteLine(string.Join(" ", focus.Select(f => f.ToString("F2"))));

        int maxIndex = focus.IndexOf(focus.Max());
        Console.WriteLine($"Max at {maxIndex}/{focus.Count} ({100 * maxIndex / focus.Count}%)");

        List<Mat> imageset = images.Where((image, index) => focus[index] >= threshold).ToList();

        if (imageset.Count < 5)
        {
            double maxFv = focus.Max() * 0.66;
            imageset = images.Where((image, index) => focus[index] >= maxFv).ToList();
        }

        return imageset;
    }

    private static Mat StackImages(List<Mat> images)
    {
        // Implement the get_pyramid_fusion logic in C# and call it here.
        return PyramidFusion.GetPyramidFusion(images.ToArray());
    }

    private static void StackDir(string dirPath, string outDirPath = null, bool overwrite = false)
    {
        string dest = outDirPath == null ? $"{dirPath}.jpg" : Path.Combine(outDirPath, Path.GetFileName(dirPath) + ".jpg");

        if (!overwrite && File.Exists(dest))
            return;

        Console.WriteLine($"Reading {dirPath}");

        List<Mat> images = Directory.GetFiles(dirPath, "*.*").Select(file => Cv2.ImRead(file, ImreadModes.Unchanged)).ToList();
        images = StackFilter(images, 1.0);

        Console.WriteLine($"Stacking {images.Count} images in {dirPath}");
        Mat stacked = StackImages(images);

        Console.WriteLine($"Writing {dest}");
        Cv2.ImWrite(dest, stacked);
    }

    public static async Task Main(string[] args)
    {
        string inputSpec = args[0];
        string outDirPath = args.Length > 1 ? args[1] : null;

        if (inputSpec.Contains('*'))
        {
            int processes = args.Length > 2 ? int.Parse(args[2]) : 4;

            List<Task> tasks = new List<Task>();
            foreach (string dirPath in Directory.GetDirectories(Path.GetDirectoryName(inputSpec), Path.GetFileName(inputSpec)))
            {
                if (Directory.Exists(dirPath))
                {
                    tasks.Add(Task.Run(() => StackDir(dirPath, outDirPath, true)));
                }
            }

            await Task.WhenAll(tasks);
        }
        else
        {
            StackDir(inputSpec, outDirPath, true);
        }
    }
}