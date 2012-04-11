#include "FrameProcessor.h"

FrameProcessor::FrameProcessor() : firstTime(true), frameNumber(0), duration(0), tictoc(""), frameToStop(0)
{
  std::cout << "FrameProcessor()" << std::endl;

  loadConfig();
  saveConfig();
}

FrameProcessor::~FrameProcessor()
{
  std::cout << "~FrameProcessor()" << std::endl;
}

void FrameProcessor::init()
{
  if(enablePreProcessor)
    preProcessor = new PreProcessor;

  if(enableFrameDifferenceBGS)
    frameDifference = new FrameDifferenceBGS;

  if(enableStaticFrameDifferenceBGS)
    staticFrameDifference = new StaticFrameDifferenceBGS;

  if(enableWeightedMovingMeanBGS)
    weightedMovingMean = new WeightedMovingMeanBGS;

  if(enableWeightedMovingVarianceBGS)
    weightedMovingVariance = new WeightedMovingVarianceBGS;

  if(enableMixtureOfGaussianV1BGS)
    mixtureOfGaussianV1BGS = new MixtureOfGaussianV1BGS;

  if(enableMixtureOfGaussianV2BGS)
    mixtureOfGaussianV2BGS = new MixtureOfGaussianV2BGS;

  if(enableAdaptiveBackgroundLearning)
    adaptiveBackgroundLearning = new AdaptiveBackgroundLearning;

  if(enableDPAdaptiveMedianBGS)
    adaptiveMedian = new DPAdaptiveMedianBGS;

  if(enableDPGrimsonGMMBGS)
    grimsonGMM = new DPGrimsonGMMBGS;

  if(enableDPZivkovicAGMMBGS)
    zivkovicAGMM = new DPZivkovicAGMMBGS;

  if(enableDPMeanBGS)
    temporalMean = new DPMeanBGS;

  if(enableDPWrenGABGS)
    wrenGA = new DPWrenGABGS;

  if(enableDPPratiMediodBGS)
    pratiMediod = new DPPratiMediodBGS;

  if(enableDPEigenbackgroundBGS)
    eigenBackground = new DPEigenbackgroundBGS;

  if(enableT2FGMM_UM)
    type2FuzzyGMM_UM = new T2FGMM_UM;

  if(enableT2FGMM_UV)
    type2FuzzyGMM_UV = new T2FGMM_UV;
  
  if(enableMultiLayerBGS)
    multiLayerBGS = new MultiLayerBGS;
  
  if(enableLBSimpleGaussian)
    lbSimpleGaussian = new LBSimpleGaussian;

  if(enableLBFuzzyGaussian)
    lbFuzzyGaussian = new LBFuzzyGaussian;

  if(enableLBMixtureOfGaussians)
    lbMixtureOfGaussians = new LBMixtureOfGaussians;

  if(enableLBAdaptiveSOM)
    lbAdaptiveSOM = new LBAdaptiveSOM;

  if(enableLBFuzzyAdaptiveSOM)
    lbFuzzyAdaptiveSOM = new LBFuzzyAdaptiveSOM;

  if(enableForegroundMaskAnalysis)
    foregroundMaskAnalysis = new ForegroundMaskAnalysis;
}

void FrameProcessor::process(std::string name, IBGS *bgs, const cv::Mat &img_input, cv::Mat &img_bgs)
{
  if(tictoc == name)
    tic(name);

  bgs->process(img_input, img_bgs);

  if(tictoc == name)
    toc();
}

void FrameProcessor::process(const cv::Mat &img_input)
{
  frameNumber++;

  if(enablePreProcessor)
    preProcessor->process(img_input, img_prep);
  
  if(enableFrameDifferenceBGS)
    process("FrameDifferenceBGS", frameDifference, img_prep, img_framediff);
  
  if(enableStaticFrameDifferenceBGS)
    process("StaticFrameDifferenceBGS", staticFrameDifference, img_prep, img_staticfdiff);
  
  if(enableWeightedMovingMeanBGS)
    process("WeightedMovingMeanBGS", weightedMovingMean, img_prep, img_wmovmean);
  
  if(enableWeightedMovingVarianceBGS)
    process("WeightedMovingVarianceBGS", weightedMovingVariance, img_prep, img_movvar);
  
  if(enableMixtureOfGaussianV1BGS)
    process("MixtureOfGaussianV1BGS", mixtureOfGaussianV1BGS, img_prep, img_mog1);
  
  if(enableMixtureOfGaussianV2BGS)
    process("MixtureOfGaussianV2BGS", mixtureOfGaussianV2BGS, img_prep, img_mog2);
  
  if(enableAdaptiveBackgroundLearning)
    process("AdaptiveBackgroundLearning", adaptiveBackgroundLearning, img_prep, img_bkgl_fgmask);

  if(enableDPAdaptiveMedianBGS)
    process("DPAdaptiveMedianBGS", adaptiveMedian, img_prep, img_adpmed);
  
  if(enableDPGrimsonGMMBGS)
    process("DPGrimsonGMMBGS", grimsonGMM, img_prep, img_grigmm);
  
  if(enableDPZivkovicAGMMBGS)
    process("DPZivkovicAGMMBGS", zivkovicAGMM, img_prep, img_zivgmm);
  
  if(enableDPMeanBGS)
    process("DPMeanBGS", temporalMean, img_prep, img_tmpmean);
  
  if(enableDPWrenGABGS)
    process("DPWrenGABGS", wrenGA, img_prep, img_wrenga);
  
  if(enableDPPratiMediodBGS)
    process("DPPratiMediodBGS", pratiMediod, img_prep, img_pramed);
  
  if(enableDPEigenbackgroundBGS)
    process("DPEigenbackgroundBGS", eigenBackground, img_input, img_eigbkg);

  if(enableT2FGMM_UM)
    process("T2FGMM_UM", type2FuzzyGMM_UM, img_prep, img_t2fgmm_um);

  if(enableT2FGMM_UV)
    process("T2FGMM_UV", type2FuzzyGMM_UV, img_prep, img_t2fgmm_uv);

  if(enableMultiLayerBGS)
  {
    multiLayerBGS->setStatus(MultiLayerBGS::Status::MLBGS_LEARN);
    //multiLayerBGS->setStatus(MultiLayerBGS::Status::MLBGS_DETECT);
    process("MultiLayerBGS", multiLayerBGS, img_input, img_mlbgs);
  }
  
  if(enableLBSimpleGaussian)
    process("LBSimpleGaussian", lbSimpleGaussian, img_input, img_lb_sg);
  
  if(enableLBFuzzyGaussian)
    process("LBFuzzyGaussian", lbFuzzyGaussian, img_input, img_lb_fg);

  if(enableLBMixtureOfGaussians)
    process("LBMixtureOfGaussians", lbMixtureOfGaussians, img_input, img_lb_mog);

  if(enableLBAdaptiveSOM)
    process("LBAdaptiveSOM", lbAdaptiveSOM, img_input, img_lb_som);

  if(enableLBFuzzyAdaptiveSOM)
    process("LBFuzzyAdaptiveSOM", lbFuzzyAdaptiveSOM, img_input, img_lb_fsom);

  if(enableForegroundMaskAnalysis)
  {
    foregroundMaskAnalysis->stopAt = frameToStop;
    foregroundMaskAnalysis->img_ref_path = imgref;

    foregroundMaskAnalysis->process(frameNumber, "FrameDifferenceBGS", img_framediff);
    foregroundMaskAnalysis->process(frameNumber, "StaticFrameDifferenceBGS", img_staticfdiff);
    foregroundMaskAnalysis->process(frameNumber, "WeightedMovingMeanBGS", img_wmovmean);
    foregroundMaskAnalysis->process(frameNumber, "WeightedMovingVarianceBGS", img_movvar);
    foregroundMaskAnalysis->process(frameNumber, "MixtureOfGaussianV1BGS", img_mog1);
    foregroundMaskAnalysis->process(frameNumber, "MixtureOfGaussianV2BGS", img_mog2);
    foregroundMaskAnalysis->process(frameNumber, "AdaptiveBackgroundLearning", img_bkgl_fgmask);
    foregroundMaskAnalysis->process(frameNumber, "DPAdaptiveMedianBGS", img_adpmed);
    foregroundMaskAnalysis->process(frameNumber, "DPGrimsonGMMBGS", img_grigmm);
    foregroundMaskAnalysis->process(frameNumber, "DPZivkovicAGMMBGS", img_zivgmm);
    foregroundMaskAnalysis->process(frameNumber, "DPMeanBGS", img_tmpmean);
    foregroundMaskAnalysis->process(frameNumber, "DPWrenGABGS", img_wrenga);
    foregroundMaskAnalysis->process(frameNumber, "DPPratiMediodBGS", img_pramed);
    foregroundMaskAnalysis->process(frameNumber, "DPEigenbackgroundBGS", img_eigbkg);
    foregroundMaskAnalysis->process(frameNumber, "T2FGMM_UM", img_t2fgmm_um);
    foregroundMaskAnalysis->process(frameNumber, "T2FGMM_UV", img_t2fgmm_uv);
    foregroundMaskAnalysis->process(frameNumber, "MultiLayerBGS", img_mlbgs);
    foregroundMaskAnalysis->process(frameNumber, "LBSimpleGaussian", img_lb_sg);
    foregroundMaskAnalysis->process(frameNumber, "LBFuzzyGaussian", img_lb_fg);
    foregroundMaskAnalysis->process(frameNumber, "LBMixtureOfGaussians", img_lb_mog);
    foregroundMaskAnalysis->process(frameNumber, "LBAdaptiveSOM", img_lb_som);
    foregroundMaskAnalysis->process(frameNumber, "LBFuzzyAdaptiveSOM", img_lb_fsom);
  }

  firstTime = false;
}

void FrameProcessor::finish(void)
{
  if(enableMultiLayerBGS)
    multiLayerBGS->finish();

  if(enableLBSimpleGaussian)
    lbSimpleGaussian->finish();
  
  if(enableLBFuzzyGaussian)
    lbFuzzyGaussian->finish();

  if(enableLBMixtureOfGaussians)
    lbMixtureOfGaussians->finish();

  if(enableLBAdaptiveSOM)
    lbAdaptiveSOM->finish();

  if(enableLBFuzzyAdaptiveSOM)
    lbFuzzyAdaptiveSOM->finish();
  
  if(enableForegroundMaskAnalysis)
    delete foregroundMaskAnalysis;
  
  if(enableLBFuzzyAdaptiveSOM)
    delete lbFuzzyAdaptiveSOM;

  if(enableLBAdaptiveSOM)
    delete lbAdaptiveSOM;

  if(enableLBMixtureOfGaussians)
    delete lbMixtureOfGaussians;

  if(enableLBFuzzyGaussian)
    delete lbFuzzyGaussian;

  if(enableLBSimpleGaussian)
    delete lbSimpleGaussian;
  
  if(enableMultiLayerBGS)
    delete multiLayerBGS;

  if(enableT2FGMM_UV)
    delete type2FuzzyGMM_UV;

  if(enableT2FGMM_UM)
    delete type2FuzzyGMM_UM;

  if(enableDPEigenbackgroundBGS)
    delete eigenBackground;

  if(enableDPPratiMediodBGS)
    delete pratiMediod;

  if(enableDPWrenGABGS)
    delete wrenGA;

  if(enableDPMeanBGS)
    delete temporalMean;

  if(enableDPZivkovicAGMMBGS)
    delete zivkovicAGMM;

  if(enableDPGrimsonGMMBGS)
    delete grimsonGMM;

  if(enableDPAdaptiveMedianBGS)
    delete adaptiveMedian;

  if(enableAdaptiveBackgroundLearning)
    delete adaptiveBackgroundLearning;

  if(enableMixtureOfGaussianV2BGS)
    delete mixtureOfGaussianV2BGS;

  if(enableMixtureOfGaussianV1BGS)
    delete mixtureOfGaussianV1BGS;

  if(enableWeightedMovingVarianceBGS)
    delete weightedMovingVariance;

  if(enableWeightedMovingMeanBGS)
    delete weightedMovingMean;

  if(enableStaticFrameDifferenceBGS)
    delete staticFrameDifference;

  if(enableFrameDifferenceBGS)
    delete frameDifference;

  if(enablePreProcessor)
    delete preProcessor;
}

void FrameProcessor::tic(std::string value)
{
  processname = value;
  duration = static_cast<double>(cv::getTickCount());
}

void FrameProcessor::toc()
{
  duration = (static_cast<double>(cv::getTickCount()) - duration)/cv::getTickFrequency();
  std::cout << processname << "\ttime(sec):" << std::fixed << std::setprecision(6) << duration << std::endl;
}

void FrameProcessor::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_WRITE);

  cvWriteString(fs, "tictoc", tictoc.c_str());

  cvWriteInt(fs, "enablePreProcessor", enablePreProcessor);
  
  cvWriteInt(fs, "enableForegroundMaskAnalysis", enableForegroundMaskAnalysis);

  cvWriteInt(fs, "enableFrameDifferenceBGS", enableFrameDifferenceBGS);
  cvWriteInt(fs, "enableStaticFrameDifferenceBGS", enableStaticFrameDifferenceBGS);
  cvWriteInt(fs, "enableWeightedMovingMeanBGS", enableWeightedMovingMeanBGS);
  cvWriteInt(fs, "enableWeightedMovingVarianceBGS", enableWeightedMovingVarianceBGS);
  cvWriteInt(fs, "enableMixtureOfGaussianV1BGS", enableMixtureOfGaussianV1BGS);
  cvWriteInt(fs, "enableMixtureOfGaussianV2BGS", enableMixtureOfGaussianV2BGS);
  cvWriteInt(fs, "enableAdaptiveBackgroundLearning", enableAdaptiveBackgroundLearning);
  
  cvWriteInt(fs, "enableDPAdaptiveMedianBGS", enableDPAdaptiveMedianBGS);
  cvWriteInt(fs, "enableDPGrimsonGMMBGS", enableDPGrimsonGMMBGS);
  cvWriteInt(fs, "enableDPZivkovicAGMMBGS", enableDPZivkovicAGMMBGS);
  cvWriteInt(fs, "enableDPMeanBGS", enableDPMeanBGS);
  cvWriteInt(fs, "enableDPWrenGABGS", enableDPWrenGABGS);
  cvWriteInt(fs, "enableDPPratiMediodBGS", enableDPPratiMediodBGS);
  cvWriteInt(fs, "enableDPEigenbackgroundBGS", enableDPEigenbackgroundBGS);

  cvWriteInt(fs, "enableT2FGMM_UM", enableT2FGMM_UM);
  cvWriteInt(fs, "enableT2FGMM_UV", enableT2FGMM_UV);

  cvWriteInt(fs, "enableMultiLayerBGS", enableMultiLayerBGS);

  cvWriteInt(fs, "enableLBSimpleGaussian", enableLBSimpleGaussian);
  cvWriteInt(fs, "enableLBFuzzyGaussian", enableLBFuzzyGaussian);
  cvWriteInt(fs, "enableLBMixtureOfGaussians", enableLBMixtureOfGaussians);
  cvWriteInt(fs, "enableLBAdaptiveSOM", enableLBAdaptiveSOM);
  cvWriteInt(fs, "enableLBFuzzyAdaptiveSOM", enableLBFuzzyAdaptiveSOM);

  cvReleaseFileStorage(&fs);
}

void FrameProcessor::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_READ);
  
  tictoc = cvReadStringByName(fs, 0, "tictoc", "");

  enablePreProcessor = cvReadIntByName(fs, 0, "enablePreProcessor", true);

  enableForegroundMaskAnalysis = cvReadIntByName(fs, 0, "enableForegroundMaskAnalysis", false);
  
  enableFrameDifferenceBGS = cvReadIntByName(fs, 0, "enableFrameDifferenceBGS", false);
  enableStaticFrameDifferenceBGS = cvReadIntByName(fs, 0, "enableStaticFrameDifferenceBGS", false);
  enableWeightedMovingMeanBGS = cvReadIntByName(fs, 0, "enableWeightedMovingMeanBGS", false);
  enableWeightedMovingVarianceBGS = cvReadIntByName(fs, 0, "enableWeightedMovingVarianceBGS", false);
  enableMixtureOfGaussianV1BGS = cvReadIntByName(fs, 0, "enableMixtureOfGaussianV1BGS", false);
  enableMixtureOfGaussianV2BGS = cvReadIntByName(fs, 0, "enableMixtureOfGaussianV2BGS", false);
  enableAdaptiveBackgroundLearning = cvReadIntByName(fs, 0, "enableAdaptiveBackgroundLearning", false);

  enableDPAdaptiveMedianBGS = cvReadIntByName(fs, 0, "enableDPAdaptiveMedianBGS", false);
  enableDPGrimsonGMMBGS = cvReadIntByName(fs, 0, "enableDPGrimsonGMMBGS", false);
  enableDPZivkovicAGMMBGS = cvReadIntByName(fs, 0, "enableDPZivkovicAGMMBGS", false);
  enableDPMeanBGS = cvReadIntByName(fs, 0, "enableDPMeanBGS", false);
  enableDPWrenGABGS = cvReadIntByName(fs, 0, "enableDPWrenGABGS", false);
  enableDPPratiMediodBGS = cvReadIntByName(fs, 0, "enableDPPratiMediodBGS", false);
  enableDPEigenbackgroundBGS = cvReadIntByName(fs, 0, "enableDPEigenbackgroundBGS", false);

  enableT2FGMM_UM = cvReadIntByName(fs, 0, "enableT2FGMM_UM", false);
  enableT2FGMM_UV = cvReadIntByName(fs, 0, "enableT2FGMM_UV", false);

  enableMultiLayerBGS = cvReadIntByName(fs, 0, "enableMultiLayerBGS", false);

  enableLBSimpleGaussian = cvReadIntByName(fs, 0, "enableLBSimpleGaussian", false);
  enableLBFuzzyGaussian = cvReadIntByName(fs, 0, "enableLBFuzzyGaussian", false);
  enableLBMixtureOfGaussians = cvReadIntByName(fs, 0, "enableLBMixtureOfGaussians", false);
  enableLBAdaptiveSOM = cvReadIntByName(fs, 0, "enableLBAdaptiveSOM", false);
  enableLBFuzzyAdaptiveSOM = cvReadIntByName(fs, 0, "enableLBFuzzyAdaptiveSOM", false);

  cvReleaseFileStorage(&fs);
}