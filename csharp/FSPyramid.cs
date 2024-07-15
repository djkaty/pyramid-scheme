using System;
using System.Linq;
using OpenCvSharp;

public class PyramidFusion
{
    private const float INTERNAL_DTYP = (float)typeof(float);

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

    public static Mat Convolve(Mat image, Mat kernel)
    {
        Mat result = new Mat();
        Cv2.Filter2D(image, result, -1, kernel, borderType: BorderTypes.Reflect101);
        return result;
    }

    public static Mat[] GaussianPyramid(Mat[] images, int levels)
    {
        var pyramid = new Mat[levels + 1];
        pyramid[0] = images[0];

        for (int i = 1; i <= levels; i++)
        {
            Mat[] level = new Mat[images.Length];
            for (int j = 0; j < images.Length; j++)
            {
                level[j] = new Mat();
                Cv2.PyrDown(pyramid[i - 1][j], level[j]);
            }
            pyramid[i] = new Mat(level);
        }

        return pyramid;
    }

    public static Mat[] LaplacianPyramid(Mat[] images, int levels)
    {
        var gaussian = GaussianPyramid(images, levels);
        var pyramid = new Mat[levels + 1];
        pyramid[levels] = gaussian[levels];

        for (int level = levels - 1; level >= 0; level--)
        {
            Mat[] levelArray = new Mat[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                Mat expanded = new Mat();
                Cv2.PyrUp(gaussian[level + 1][i], expanded);

                if (expanded.Size() != gaussian[level][i].Size())
                {
                    expanded = expanded[0, gaussian[level][i].Rows, 0, gaussian[level][i].Cols];
                }

                levelArray[i] = gaussian[level][i] - expanded;
            }
            pyramid[level] = new Mat(levelArray);
        }

        Array.Reverse(pyramid);
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

    public static float[] GetProbabilities(Mat grayImage)
    {
        var histogram = new int[256];
        grayImage.ForEachAsByte((value, pos) => histogram[value]++);

        float totalPixels = grayImage.Rows * grayImage.Cols;
        var probabilities = new float[256];
        for (int i = 0; i < 256; i++)
        {
            probabilities[i] = histogram[i] / totalPixels;
        }

        return probabilities;
    }

    public static Tuple<Mat, Mat> EntropyDeviation(Mat image, int kernelSize)
    {
        float[] probabilities = GetProbabilities(image);
        int padAmount = (kernelSize - 1) / 2;
        Mat paddedImage = new Mat();
        Cv2.CopyMakeBorder(image, paddedImage, padAmount, padAmount, padAmount, padAmount, BorderTypes.Reflect101);

        Mat entropies = new Mat(image.Size(), MatType.CV_32F);
        Mat deviations = new Mat(image.Size(), MatType.CV_32F);

        for (int row = 0; row < image.Rows; row++)
        {
            for (int column = 0; column < image.Cols; column++)
            {
                Mat area = paddedImage.SubMat(row, row + kernelSize, column, column + kernelSize);
                entropies.Set<float>(row, column, -area.ToBytes().Sum(b => probabilities[b] * Math.Log(probabilities[b])));
                deviations.Set<float>(row, column, (float)(area.ToBytes().Average() - area.ToBytes().Sum(b => Math.Pow(b - area.ToBytes().Average(), 2) / area.Total())));
            }
        }

        return Tuple.Create(entropies, deviations);
    }

    public static Mat GetFusedBaseChannel(Mat[] images, int kernelSize, int channel)
    {
        Mat entropies = new Mat(images[0].Size(), MatType.CV_32F);
        Mat deviations = new Mat(images[0].Size(), MatType.CV_32F);

        for (int i = 0; i < images.Length; i++)
        {
            Mat grayImage = new Mat();
            Cv2.CvtColor(images[i], grayImage, ColorConversionCodes.BGR2GRAY);
            var result = EntropyDeviation(grayImage, kernelSize);
            entropies += result.Item1;
            deviations += result.Item2;
        }

        var fused = new Mat(images[0].Size(), images[0].Type());
        for (int i = 0; i < images.Length; i++)
        {
            fused += images[i];
        }

        return fused / 2;
    }

    public static Mat GetFusedBase(Mat[] images, int kernelSize)
    {
        Mat fused = new Mat(images[0].Size(), images[0].Type());
        for (int channel = 0; channel < images[0].Channels(); channel++)
        {
            fused.SetArray(channel, GetFusedBaseChannel(images, kernelSize, channel));
        }
        return fused;
    }

    public static Mat[] FusePyramids(Mat[] pyramids, int kernelSize)
    {
        Mat fused = GetFusedBase(pyramids[pyramids.Length - 1], kernelSize);
        for (int i = pyramids.Length - 2; i >= 0; i--)
        {
            fused += pyramids[i];
        }
        return new[] { fused };
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