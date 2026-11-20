using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace MuscleThicknessApp
{
    public partial class MainWindow : Window
    {
        const int IMG_SIZE = 640;
        const double PX_TO_MM = 0.058;   // ✅ CHANGE this if calibration differs

        // The model files are copied next to the executable at build time (see MuscleThicknessApp.csproj).
        string OnnxPath = System.IO.Path.Combine(AppContext.BaseDirectory, "apo_model.onnx");

        InferenceSession session;

        byte[,]? gray;
        int[,]? mask;

        int W, H;

        bool drawing = false;
        Point start;
        Rect roi;

        public MainWindow()
        {
            InitializeComponent();
            session = new InferenceSession(OnnxPath);
            StatusText.Text = "Model loaded.";
        }

        // =========================
        // LOAD IMAGE
        // =========================
        private void BtnLoadImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            if (dlg.ShowDialog() != true) return;

            BitmapImage bmp = new BitmapImage(new Uri(dlg.FileName));
            FormatConvertedBitmap grayBmp = new FormatConvertedBitmap(
                bmp, System.Windows.Media.PixelFormats.Gray8, null, 0);

            W = grayBmp.PixelWidth;
            H = grayBmp.PixelHeight;

            byte[] raw = new byte[W * H];
            grayBmp.CopyPixels(raw, W, 0);

            gray = new byte[H, W];

            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    gray[y, x] = raw[y * W + x];

            MainImage.Source = grayBmp;
            RoiRect.Visibility = Visibility.Collapsed;

            ThicknessText.Text = "-- px (-- mm)";
            StatusText.Text = "Image loaded.";
        }

        // =========================
        // RUN SEGMENTATION
        // =========================
        private void BtnSegment_Click(object sender, RoutedEventArgs e)
        {
            if (gray == null)
            {
                StatusText.Text = "Load image first.";
                return;
            }

            float[] flat = new float[IMG_SIZE * IMG_SIZE];

            for (int y = 0; y < IMG_SIZE; y++)
            {
                int sy = y * H / IMG_SIZE;
                for (int x = 0; x < IMG_SIZE; x++)
                {
                    int sx = x * W / IMG_SIZE;
                    flat[y * IMG_SIZE + x] = gray[sy, sx] / 255f;
                }
            }

            var tensor = new DenseTensor<float>(flat,
                new int[] { 1, 1, IMG_SIZE, IMG_SIZE });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();  // (1,3,640,640)

            int[,] small = new int[IMG_SIZE, IMG_SIZE];

            for (int y = 0; y < IMG_SIZE; y++)
            {
                for (int x = 0; x < IMG_SIZE; x++)
                {
                    int cls = 0;
                    float best = output[0, 0, y, x];

                    for (int c = 1; c < 3; c++)
                        if (output[0, c, y, x] > best)
                        {
                            best = output[0, c, y, x];
                            cls = c;
                        }

                    small[y, x] = cls;
                }
            }

            mask = new int[H, W];

            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    mask[y, x] = small[y * IMG_SIZE / H, x * IMG_SIZE / W];

            DrawOverlay();
            StatusText.Text = "Segmentation complete.";
        }

        // =========================
        // OVERLAY DRAW
        // =========================
        private void DrawOverlay()
        {
            if (gray == null || mask == null) return;

            int stride = W * 4;
            byte[] pix = new byte[stride * H];
            double alpha = 0.5;

            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    byte g = gray[y, x];
                    double r = g, gr = g, b = g;

                    if (mask[y, x] == 1) { r = 255; gr = 0; b = 0; }
                    if (mask[y, x] == 2) { r = 255; gr = 255; b = 0; }

                    int i = y * stride + x * 4;
                    pix[i] = (byte)((1 - alpha) * g + alpha * b);
                    pix[i + 1] = (byte)((1 - alpha) * g + alpha * gr);
                    pix[i + 2] = (byte)((1 - alpha) * g + alpha * r);
                    pix[i + 3] = 255;
                }

            MainImage.Source = BitmapSource.Create(W, H, 96, 96,
                System.Windows.Media.PixelFormats.Bgra32,
                null, pix, stride);
        }

        // =========================
        // ROI FIX (NO OFFSET BUG)
        // =========================
        private void RoiCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            drawing = true;
            start = e.GetPosition(RoiCanvas);

            RoiRect.Visibility = Visibility.Visible;
        }

        private void RoiCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (!drawing) return;

            Point p = e.GetPosition(RoiCanvas);

            double x = Math.Min(start.X, p.X);
            double y = Math.Min(start.Y, p.Y);
            double w = Math.Abs(start.X - p.X);
            double h = Math.Abs(start.Y - p.Y);

            roi = new Rect(x, y, w, h);

            Canvas.SetLeft(RoiRect, x);
            Canvas.SetTop(RoiRect, y);
            RoiRect.Width = w;
            RoiRect.Height = h;
        }

        private void RoiCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            drawing = false;
        }

        // =========================
        // THICKNESS (PX + MM)
        // =========================
        private void BtnThickness_Click(object sender, RoutedEventArgs e)
        {
            if (mask == null || roi.Width < 2)
            {
                StatusText.Text = "Run segmentation and draw ROI.";
                return;
            }

            double px = MeasureThickness();
            double mm = px * PX_TO_MM;

            ThicknessText.Text = $"{px:0.0} px ({mm:0.00} mm)";
            StatusText.Text = "Thickness computed.";
        }

        private double MeasureThickness()
        {
            int x1 = Math.Max(0, (int)roi.X);
            int x2 = Math.Min(W - 1, (int)(roi.X + roi.Width));
            int y1 = Math.Max(0, (int)roi.Y);
            int y2 = Math.Min(H - 1, (int)(roi.Y + roi.Height));

            List<int> supY = new List<int>();
            List<int> deepY = new List<int>();

            for (int y = y1; y < y2; y++)
            {
                for (int x = x1; x < x2; x++)
                {
                    if (mask[y, x] == 1) supY.Add(y);
                    if (mask[y, x] == 2) deepY.Add(y);
                }
            }

            if (supY.Count == 0 || deepY.Count == 0)
                return 0;

            double meanSup = supY.Average();
            double meanDeep = deepY.Average();

            return Math.Abs(meanDeep - meanSup);
        }

    }
}


