using OpenCvSharp;

public class PyramidFusion
{
    private const int kernelSize = 5;

    private static Mat<float> GenerateKernel(float a)
    {
        float[] kernel = [0.25f - a / 2.0f, 0.25f, a, 0.25f, 0.25f - a / 2.0f];
        var kernelMat = new Mat<float>(kernel.Length, kernel.Length);

        for (int i = 0; i < kernel.Length; i++) {
            for (int j = 0; j < kernel.Length; j++) {
                kernelMat.Set(i, j, kernel[i] * kernel[j]);
            }
        }
        return kernelMat;
    }

    private static Mat Convolve(Mat image, Mat? kernel = null)
    {
        kernel ??= GenerateKernel(0.4f);
        var result = new Mat();
        Cv2.Filter2D(image, result, -1, kernel, borderType: BorderTypes.Reflect101);
        return result;
    }

    private static Mat[][] GaussianPyramid(Mat[] images, int depth)
    {
        var pyramid = new Mat[depth + 1][];
        pyramid[0] = images;

        for (int level = 1; level <= depth; level++) {
            var pyramidLevel = new Mat[images.Length];
            for (int image = 0; image < images.Length; image++) {
                pyramidLevel[image] = new Mat();
                Cv2.PyrDown(pyramid[level - 1][image], pyramidLevel[image]);
            }
            pyramid[level] = pyramidLevel;
        }
        return pyramid;
    }

    private static Mat[][] LaplacianPyramid(Mat[] images, int depth)
    {
        var gaussian = GaussianPyramid(images, depth);
        var pyramid = new Mat[gaussian.Length][];
        pyramid[^1] = gaussian[^1];

        for (int level = gaussian.Length - 1; level > 0; level--) {
            var gaussianImagesForLevel = gaussian[level - 1];
            var imagesForLevel = new Mat[images.Length];
            for (int image = 0; image < images.Length; image++) {
                var gaussianImage = gaussianImagesForLevel[image];
                var expanded = new Mat();
                Cv2.PyrUp(gaussian[level][image], expanded);

                if (expanded.Size() != gaussianImage.Size()) {
                    expanded = expanded[0, gaussianImage.Rows, 0, gaussianImage.Cols];
                }
                imagesForLevel[image] = gaussianImage - expanded;
            }
            pyramid[level - 1] = imagesForLevel;
        }
        return pyramid;
    }

    private static Mat Collapse(Mat[] pyramid)
    {
        var image = pyramid[^1];
        for (int i = pyramid.Length - 2; i >= 0; i--) {
            var expanded = new Mat();
            Cv2.PyrUp(image, expanded);

            if (expanded.Size() != pyramid[i].Size()) {
                expanded = expanded[0, pyramid[i].Rows, 0, pyramid[i].Cols];
            }
            image = expanded + pyramid[i];
        }
        return image;
    }

    private static Mat FindDeviations(Mat image, int kernelSize)
    {
        var padAmount = (kernelSize - 1) / 2;
        var paddedImage = new Mat();
        Cv2.CopyMakeBorder(image, paddedImage, padAmount, padAmount, padAmount, padAmount, BorderTypes.Reflect101);
        paddedImage.GetArray(out float[] pixels);
        var width = paddedImage.Width;

        var deviations = new Mat(image.Size(), MatType.CV_32F);

        var averageWindow = new Queue<float>(kernelSize);
        var pixelWindowSize = kernelSize * kernelSize;

        for (int row = 0; row < image.Rows; row++)
        {
            averageWindow.Clear();
            var currentSum = 0f;
            for (int columnOffset = 0; columnOffset < kernelSize; columnOffset++) {
                var sumForColumn = 0f;
                for (int rowOffset = 0; rowOffset < kernelSize; rowOffset++) {
                    sumForColumn += pixels[(row + rowOffset) * width + columnOffset];
                }
                currentSum += sumForColumn;
                averageWindow.Enqueue(sumForColumn);
            }

            for (int column = 0; column < image.Cols; column++)
            {
                if (column > 0) {
                    currentSum -= averageWindow.Dequeue();
                    var sumForColumn = 0f;
                    for (int rowOffset = 0; rowOffset < kernelSize; rowOffset++) {
                        sumForColumn += pixels[(row + rowOffset) * width + column + kernelSize - 1];
                    }
                    currentSum += sumForColumn;
                    averageWindow.Enqueue(sumForColumn);
                }

                var average = currentSum / pixelWindowSize;
                var sum = 0f;
                for (int y = 0; y < kernelSize; y++) {
                    for (int x = 0; x < kernelSize; x++) {
                        sum += (pixels[y * width + x] - average) * (pixels[y * width + x] - average);
                    }
                }
                var deviation = sum / pixelWindowSize;
                deviations.Set(row, column, deviation);
            }
        }
        return deviations;
    }

    private static Mat GetFusedBaseChannel(Mat[] images, int kernelSize, int channel)
    {
        (int width, int height) = images[0].Size();

        var deviations = new Mat[images.Length];
        for (int layer = 0; layer < images.Length; layer++) {
            var grayImage = images[layer].ExtractChannel(channel);
            deviations[layer] = FindDeviations(grayImage, kernelSize);
        }

        var highestDeviation = new int[height, width];
        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                var highestDeviationForPixel = Enumerable.Range(0, images.Length).MaxBy(imageIndex => deviations[imageIndex].Get<float>(y, x));
                highestDeviation[y, x] = highestDeviationForPixel;
            }
        }

        var fused = new float[height, width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                fused[y, x] = images[highestDeviation[y, x]].Get<Vec3f>(y, x)[channel];
            }
        }
        return Mat.FromArray(fused);
    }

    private static Mat GetFusedBase(Mat[] images, int kernelSize)
    {
        Mat fused = new Mat<Vec3f>(images[0].Height, images[0].Width);
        for (int channel = 0; channel < images[0].Channels(); channel++) {
            GetFusedBaseChannel(images, kernelSize, channel).InsertChannel(fused, channel);
        }
        return fused;
    }

    private static Mat[] FusePyramids(Mat[][] pyramids, int kernelSize)
    {
        var fused = new Mat[pyramids.Length];
        fused[^1] = GetFusedBase(pyramids[^1], kernelSize);
        for (int i = pyramids.Length - 2; i >= 0; i--) {
            fused[i] = GetFusedLaplacian(pyramids[i]);
        }
        return fused;
    }

    private static Mat<float> GetPixelsFromBestRegionEnergies(Mat[] laplacians, int channel)
    {
        (int width, int height) = laplacians[0].Size();
        var highestRegionEnergy = new float[height * width];
        var bestPixel = new float[height * width];

        for (int layer = 0; layer < laplacians.Length; layer++) {
            var greyScale = laplacians[layer].ExtractChannel(channel);
            greyScale.GetArray(out float[] singleChannel);
            GetRegionEnergy(greyScale).GetArray(out float[] regionEnergies);
            for (int i = 0; i < height * width; i++) {
                if (regionEnergies[i] > highestRegionEnergy[i]) {
                    highestRegionEnergy[i] = regionEnergies[i];
                    bestPixel[i] = singleChannel[i];
                }
            }
        }
        return Mat.FromArray(bestPixel).Reshape(height, width);
    }

    private static Mat GetFusedLaplacian(Mat[] laplacians)
    {
        (int width, int height) = laplacians[0].Size();
        int channels = laplacians[0].Channels(); // should always be 3
        var fused = new Mat<Vec3f>(height, width);
        for (int channel = 0; channel < channels; channel++) {
            var bestRegionEnergyPixels = GetPixelsFromBestRegionEnergies(laplacians, channel);
            bestRegionEnergyPixels.InsertChannel(fused, channel);
        }
        return fused;
    }

    private static Mat GetRegionEnergy(Mat laplacian)
    {
        return Convolve(laplacian.Pow(2));
    }

    public static Mat GetPyramidFusion(Mat[] images, int minSize = 32)
    {
        for (int i = 0; i < images.Length; i++) {
            images[i].ConvertTo(images[i], MatType.CV_32FC3);
        }
        var smallestSide = Math.Min(images[0].Rows, images[0].Cols);
        var depth = (int)Math.Log2(smallestSide / minSize);
        var pyramids = LaplacianPyramid(images, depth);
        var fusion = FusePyramids(pyramids, kernelSize);
        return Collapse(fusion);
    }
}