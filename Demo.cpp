/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <opencv2/opencv.hpp>

#include "package_bgs/FrameDifferenceBGS.h"
#include "package_bgs/StaticFrameDifferenceBGS.h"
#include "package_bgs/WeightedMovingMeanBGS.h"
#include "package_bgs/WeightedMovingVarianceBGS.h"
#include "package_bgs/MixtureOfGaussianV1BGS.h"
#include "package_bgs/MixtureOfGaussianV2BGS.h"
#include "package_bgs/AdaptiveBackgroundLearning.h"
#include "package_bgs/GMG.h"

#include "package_bgs/dp/DPAdaptiveMedianBGS.h"
#include "package_bgs/dp/DPGrimsonGMMBGS.h"
#include "package_bgs/dp/DPZivkovicAGMMBGS.h"
#include "package_bgs/dp/DPMeanBGS.h"
#include "package_bgs/dp/DPWrenGABGS.h"
#include "package_bgs/dp/DPPratiMediodBGS.h"
#include "package_bgs/dp/DPEigenbackgroundBGS.h"
#include "package_bgs/dp/DPTextureBGS.h"

#include "package_bgs/tb/T2FGMM_UM.h"
#include "package_bgs/tb/T2FGMM_UV.h"
#include "package_bgs/tb/T2FMRF_UM.h"
#include "package_bgs/tb/T2FMRF_UV.h"
#include "package_bgs/tb/FuzzySugenoIntegral.h"
#include "package_bgs/tb/FuzzyChoquetIntegral.h"

#include "package_bgs/lb/LBSimpleGaussian.h"
#include "package_bgs/lb/LBFuzzyGaussian.h"
#include "package_bgs/lb/LBMixtureOfGaussians.h"
#include "package_bgs/lb/LBAdaptiveSOM.h"
#include "package_bgs/lb/LBFuzzyAdaptiveSOM.h"

#include "package_bgs/jmo/MultiLayerBGS.h"
#include "package_bgs/pt/PixelBasedAdaptiveSegmenter.h"
#include "package_bgs/av/VuMeter.h"
#include "package_bgs/ae/KDE.h"

int main(int argc, char **argv)
{
  cv::VideoCapture capture;
  int resize_factor = 50;

  if(argc > 1)
  {
    std::cout << "Opening video " << argv[1] << std::endl;
    capture.open(argv[1]);
  }
  else
  {
      capture.open(0);  // Default camera
  }

  if(!capture.isOpened())
  {
    std::cerr << "Cannot initialize video!" << std::endl;
    return 1;
  }
  
  cv::Mat frame_aux;
  cv::Mat frame;
  //IplImage *frame_aux = cvQueryFrame(capture);
  //IplImage *frame = cvCreateImage(cvSize((int)((frame_aux->width*resize_factor)/100) , (int)((frame_aux->height*resize_factor)/100)), frame_aux->depth, frame_aux->nChannels);
  capture >> frame_aux;
  cv::resize(frame_aux, frame, cv::Size(0, 0),  resize_factor/100.0, resize_factor/100.0);
  //cvResize(frame_aux, frame);

  /* Background Subtraction Methods */
  IBGS *bgs;

  /*** Default Package ***/
  //bgs = new FrameDifferenceBGS;
  //bgs = new StaticFrameDifferenceBGS;
  //bgs = new WeightedMovingMeanBGS;
  //bgs = new WeightedMovingVarianceBGS;
  //bgs = new MixtureOfGaussianV1BGS;
  bgs = new MixtureOfGaussianV2BGS;
  //bgs = new AdaptiveBackgroundLearning;
  //bgs = new GMG;
  
  /*** DP Package (adapted from Donovan Parks) ***/
  //bgs = new DPAdaptiveMedianBGS;
  //bgs = new DPGrimsonGMMBGS;
  //bgs = new DPZivkovicAGMMBGS;
  //bgs = new DPMeanBGS;
  //bgs = new DPWrenGABGS;
  //bgs = new DPPratiMediodBGS;
  //bgs = new DPEigenbackgroundBGS;
  //bgs = new DPTextureBGS;

  /*** TB Package (adapted from Thierry Bouwmans) ***/
  //bgs = new T2FGMM_UM;
  //bgs = new T2FGMM_UV;
  //bgs = new T2FMRF_UM;
  //bgs = new T2FMRF_UV;
  //bgs = new FuzzySugenoIntegral;
  //bgs = new FuzzyChoquetIntegral;

  /*** JMO Package (adapted from Jean-Marc Odobez) ***/
  //bgs = new MultiLayerBGS;

  /*** PT Package (adapted from Hofmann) ***/
  //bgs = new PixelBasedAdaptiveSegmenter;

  /*** LB Package (adapted from Laurence Bender) ***/
  //bgs = new LBSimpleGaussian;
  //bgs = new LBFuzzyGaussian;
  //bgs = new LBMixtureOfGaussians;
  //bgs = new LBAdaptiveSOM;
  //bgs = new LBFuzzyAdaptiveSOM;

  /*** AV Package (adapted from Antoine Vacavant) ***/
  //bgs = new VuMeter;

  /*** EG Package (adapted from Ahmed Elgammal) ***/
  //bgs = new KDE;

  int key = 0;
  while(key != 'q')
  {
    //frame_aux = cvQueryFrame(capture);
    capture >> frame_aux;

    cv::resize(frame_aux, frame, cv::Size(0, 0),  resize_factor/100.0, resize_factor/100.0);
    
    //cv::Mat img_input(frame);
    cv::imshow("input", frame);

    cv::Mat img_mask;
    cv::Mat img_bkgmodel;
    bgs->process(frame, img_mask, img_bkgmodel); // automatically shows the foreground mask image
    
    //if(!img_mask.empty())
    //  do something
    
    key = cvWaitKey(1);
  }

  delete bgs;

  cvDestroyAllWindows();
  capture.release();
  
  return 0;
}
