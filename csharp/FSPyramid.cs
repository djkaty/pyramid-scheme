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
        // NOTE: We replace numpy.convolve with opencv.filter2d to remove dependency on numpy
        Cv2.Filter2D(image, result, -1, kernel, borderType: BorderTypes.Reflect101);
        return result;
    }

    // Returns a guassian pyramid, each layer of which is a series of images of decreasing size
    // The bottom layer of the pyramid will be the original set of images,
    // so the resulting pyramid will have depth + 1 layers
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

    // Creates a laplacian pyramid, each layer of which is a series of images of decreasing size
    // The pyramid will have the number of layers specified in 'depth'
    private static Mat[][] LaplacianPyramid(Mat[] images, int depth)
    {
        var gaussian = GaussianPyramid(images, depth);
        var pyramid = new Mat[gaussian.Length][];
        pyramid[^1] = gaussian[^1];

        // NOTE: This is reversed from the original to avoid having to reverse the array at the end
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

    // Collapses a pyramid which has been fused down to one image per layer
    // into a single final image
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

    // Sweeps across a greyscale (single channel) image using a rectangular window of kernelSize * kernelSize pixels
    // Find the pixel in each window which has the maximum brightness deviation from the average brightness
    // of all pixels in the window. Store each maximum deviation in array

    // The edges of the image are padded so that the corners and edges can be swept without an out of bounds condition

    // NOTE: This was one of the slowest functions in the code. Optimizations:
    //   - Entropy calculations were removed as they had little-to-no effect on the result
    //   - Probability calculations were removed as they were only used by entropy calculations
    //   - The image data is handled directly in managed space
    //   - The entire image is marshalled once and accessed by offset, instead of repeating for every window
    //   - Multi-dimensional arrays are replaced with single-dimension to improve performance in .NET
    //   - Averages are calculated using a rolling sum window rather than from scratch for each pixel
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

    // Get deviation maps for a series of images
    // Create a single image by selecting each pixel from the one in the series
    // which has the highest deviation
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

    // Get a series of greyscale (single channel) images from the supplied image set
    // Create a new greyscale image by selecting the pixel from the image set that
    // has the highest region energy
    // NOTE: numpy.argmax is rolled into the main loop to avoid unnecessary nesting
    private static Mat<float> GetPixelsFromBestRegionEnergies(Mat[] images, int channel)
    {
        (int width, int height) = images[0].Size();
        var highestRegionEnergy = new float[height * width];
        var highestEnergyPixels = new float[height * width];

        for (int layer = 0; layer < images.Length; layer++) {
            var greyScale = images[layer].ExtractChannel(channel);
            greyScale.GetArray(out float[] singleChannel);
            GetRegionEnergy(greyScale).GetArray(out float[] regionEnergies);
            for (int i = 0; i < height * width; i++) {
                if (regionEnergies[i] > highestRegionEnergy[i]) {
                    highestRegionEnergy[i] = regionEnergies[i];
                    highestEnergyPixels[i] = singleChannel[i];
                }
            }
        }
        return Mat.FromArray(highestEnergyPixels).Reshape(height, width);
    }

    private static Mat GetRegionEnergy(Mat laplacian)
    {
        return Convolve(laplacian.Pow(2));
    }

    // Split a series of images into per-channel greyscale images
    // Create a single image per channel using the region energy technique
    // Merge the three channels together to produce a single final image
    private static Mat GetFusedLaplacian(Mat[] images)
    {
        var fused = new Mat<Vec3f>(images[0].Height, images[0].Width);
        for (int channel = 0; channel < images[0].Channels(); channel++) {
            var bestRegionEnergyPixels = GetPixelsFromBestRegionEnergies(images, channel);
            bestRegionEnergyPixels.InsertChannel(fused, channel);
        }
        return fused;
    }

    // Split a series of images into per-channel greyscale images
    // Create a single image per channel using the deviation technique
    // Merge the three channels together to produce a single final image
    private static Mat GetFusedBase(Mat[] images, int kernelSize)
    {
        var fused = new Mat<Vec3f>(images[0].Height, images[0].Width);
        for (int channel = 0; channel < images[0].Channels(); channel++) {
            GetFusedBaseChannel(images, kernelSize, channel).InsertChannel(fused, channel);
        }
        return fused;
    }

    // Combine each image set of each layer of the laplacian pyramid
    // into a single image per layer
    // The top (smallest) layer will be merged using the deviation technique
    // The other layers will be merged using the region energy technique
    private static Mat[] FusePyramid(Mat[][] pyramids, int kernelSize)
    {
        var fused = new Mat[pyramids.Length];
        fused[^1] = GetFusedBase(pyramids[^1], kernelSize);
        for (int i = pyramids.Length - 2; i >= 0; i--) {
            fused[i] = GetFusedLaplacian(pyramids[i]);
        }
        return fused;
    }

    // 1. Create a laplacian pyramid from a set of images, where each layer is a newly generated set of images
    // 2. Merge the images on the top layer on a per-channel basis by selecting each pixel from each image channel with the highest deviation
    // 3. Merge the images on the other layers on a per-channel basis by selecting each pixel from each image channel with the highest region energy
    // 4. Merge the single-image-per-layer pyramid
    // The images and output will be converted to 32-bit 3-channel float format
    public static Mat GetPyramidFusion(Mat[] images, int minSize = 32)
    {
        for (int i = 0; i < images.Length; i++) {
            images[i].ConvertTo(images[i], MatType.CV_32FC3);
        }
        var smallestSide = Math.Min(images[0].Rows, images[0].Cols);
        var depth = (int)Math.Log2(smallestSide / minSize);
        var pyramid = LaplacianPyramid(images, depth);
        var fusion = FusePyramid(pyramid, kernelSize);
        return Collapse(fusion);
    }
}