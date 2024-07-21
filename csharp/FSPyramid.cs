using OpenCvSharp;

public class PyramidFusion
{
    public static Mat GeneratingKernel(float a)
    {
        float[] kernel = { 0.25f - a / 2.0f, 0.25f, a, 0.25f, 0.25f - a / 2.0f };
        Mat kernelMat = new Mat(kernel.Length, kernel.Length, MatType.CV_32F);

        for (int i = 0; i < kernel.Length; i++)
        {
            for (int j = 0; j < kernel.Length; j++)
            {
                kernelMat.Set<float>(i, j, kernel[i] * kernel[j]);
            }
        }

        return kernelMat;
    }

    public static Mat Convolve(Mat image, Mat? kernel = null)
    {
        kernel ??= GeneratingKernel(0.4f);
        Mat result = new Mat();
        Cv2.Filter2D(image, result, -1, kernel, borderType: BorderTypes.Reflect101);
        return result;
    }

    public static Mat[][] GaussianPyramid(Mat[] images, int levels)
    {
        var pyramid = new Mat[levels + 1][];

        for (int i = 0; i < images.Length; i++) {
            images[i].ConvertTo(images[i], MatType.CV_32FC3);
        }
        pyramid[0] = images;

        for (int i = 1; i <= levels; i++)
        {
            Mat[] level = new Mat[images.Length];
            for (int j = 0; j < images.Length; j++)
            {
                level[j] = new Mat();
                Cv2.PyrDown(pyramid[i - 1][j], level[j]);
            }
            pyramid[i] = level;
        }

        return pyramid;
    }

    public static Mat[][] LaplacianPyramid(Mat[] images, int levels)
    {
        var gaussian = GaussianPyramid(images, levels);
        var pyramid = new Mat[gaussian.Length][];
        pyramid[gaussian.Length - 1] = gaussian[gaussian.Length - 1];

        for (int level = gaussian.Length - 1; level > 0; level--)
        {
            var gauss = gaussian[level - 1];
            Mat[] levelArray = new Mat[images.Length];
            for (int layer = 0; layer < images.Length; layer++)
            {
                var gauss_layer = gauss[layer];
                Mat expanded = new Mat();
                Cv2.PyrUp(gaussian[level][layer], expanded);

                if (expanded.Size() != gauss_layer.Size())
                {
                    expanded = expanded[0, gauss_layer.Rows, 0, gauss_layer.Cols];
                }

                levelArray[layer] = gauss_layer - expanded;
            }
            pyramid[level - 1] = levelArray;
        }
        return pyramid;
    }

    public static Mat Collapse(Mat[] pyramid)
    {
        Mat image = pyramid[pyramid.Length - 1];
        for (int i = pyramid.Length - 2; i >= 0; i--)
        {
            Mat expanded = new Mat();
            Cv2.PyrUp(image, expanded);

            if (expanded.Size() != pyramid[i].Size())
            {
                expanded = expanded[0, pyramid[i].Rows, 0, pyramid[i].Cols];
            }

            image = expanded + pyramid[i];
        }

        return image;
    }

    public static Dictionary<float, float> GetProbabilities(Mat grayImage)
    {
        var histogram = new Dictionary<float, int>();

        grayImage.GetArray(out float[] grayImageBytes);

        foreach (var pixel in grayImageBytes) {
            histogram[pixel] = histogram.ContainsKey(pixel) ? histogram[pixel] + 1 : 1;
        }

        float totalPixels = grayImage.Rows * grayImage.Cols;
        var probabilities = new Dictionary<float, float>();
        foreach (var (level, count) in histogram)
        {
            probabilities[level] = count / totalPixels;
        }

        return probabilities;
    }

    public static Mat Deviation(Mat image, int kernelSize)
    {
        var probabilities = GetProbabilities(image);
        int padAmount = (kernelSize - 1) / 2;
        Mat paddedImage = new Mat();
        Cv2.CopyMakeBorder(image, paddedImage, padAmount, padAmount, padAmount, padAmount, BorderTypes.Reflect101);

        Mat deviations = new Mat(image.Size(), MatType.CV_32F);

        for (int row = 0; row < image.Rows; row++)
        {
            for (int column = 0; column < image.Cols; column++)
            {
                Mat area = paddedImage.SubMat(row, row + kernelSize, column, column + kernelSize);
                area.GetArray(out float[] areaFloats);
                var average = areaFloats.Average();
                var deviation = areaFloats.Sum(x => Math.Pow(x - average, 2)) / areaFloats.Length;

                deviations.Set<float>(row, column, (float) deviation);
            }
        }

        return deviations;
    }

    public static Mat GetFusedBaseChannel(Mat[] images, int kernelSize, int channel)
    {
        var entropies = new Mat[images.Length];
        var deviations = new Mat[images.Length];

        for (int layer = 0; layer < images.Length; layer++)
        {
            var grayImage = images[layer].ExtractChannel(channel);
            deviations[layer] = Deviation(grayImage, kernelSize);
        }

        (int width, int height) = images[0].Size();

        var best_d = new int[height, width];

        for (var y = 0; y < images[0].Height; y++) {
            for (var x = 0; x < images[0].Width; x++) {
                var bestD = Enumerable.Range(0, images.Length).MaxBy(imageIndex => deviations[imageIndex].Get<float>(y, x));
                best_d[y, x] = bestD;
            }
        }

        var fused = new float[height, width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                fused[y, x] = images[best_d[y, x]].Get<Vec3f>(y, x)[channel];
            }
        }

        var fusedMat = Mat.FromArray(fused);

        return fusedMat;
    }

    public static Mat GetFusedBase(Mat[] images, int kernelSize)
    {
        Mat fused = new Mat(images[0].Size(), images[0].Type());

        for (int channel = 0; channel < images[0].Channels(); channel++)
        {
            GetFusedBaseChannel(images, kernelSize, channel).InsertChannel(fused, channel);
        }
        return fused;
    }

    public static Mat[] FusePyramids(Mat[][] pyramids, int kernelSize)
    {
        Mat[] fused = new Mat[pyramids.Length];
        fused[pyramids.Length - 1] = GetFusedBase(pyramids[pyramids.Length - 1], kernelSize);
        for (int i = pyramids.Length - 2; i >= 0; i--)
        {
            fused[i] = GetFusedLaplacian(pyramids[i]);
        }
        return fused;
    }

    public static Mat<float> GetPixelsFromBestRegionEnergies(Mat[] laplacians, int channel) {
        (int width, int height) = laplacians[0].Size();
        var bestRE = new float[height * width];
        var bestPixel = new float[height * width];

        for (int layer = 0; layer < laplacians.Length; layer++) {
            var grey = laplacians[layer].ExtractChannel(channel);
            grey.GetArray(out float[] singleChannel);
            RegionEnergy(grey).GetArray(out float[] regionEnergies);
            for (int i = 0; i < height * width; i++) {
                if (regionEnergies[i] > bestRE[i]) {
                    bestRE[i] = regionEnergies[i];
                    bestPixel[i] = singleChannel[i];
                }
            }
        }
        return Mat.FromArray(bestPixel).Reshape(height, width);
    }

    public static Mat GetFusedLaplacian(Mat[] laplacians) {
        (int width, int height) = laplacians[0].Size();
        int channels = laplacians[0].Channels(); // should always be 3

        var fused = new Mat<Vec3f>(height, width);
        for (int channel = 0; channel < channels; channel++) {
            var bestREPixels = GetPixelsFromBestRegionEnergies(laplacians, channel);
            bestREPixels.InsertChannel(fused, channel);
        }
        return fused;
    }

    public static Mat RegionEnergy(Mat laplacian) {
        return Convolve(laplacian.Pow(2));
    }

    public static Mat GetPyramidFusion(Mat[] images, int minSize = 32)
    {
        int smallestSide = Math.Min(images[0].Rows, images[0].Cols);
        int depth = (int)Math.Log2(smallestSide / minSize);
        int kernelSize = 5;

        var pyramids = LaplacianPyramid(images, depth);
        var fusion = FusePyramids(pyramids, kernelSize);
        return Collapse(fusion);
    }
}