import arcpy
import numpy
import pylab
import os
import ConversionUtils
from sklearn.metrics import roc_curve, auc
arcpy.CheckOutExtension("spatial")

inputFeature = arcpy.GetParameterAsText(0)
predict_raster_files = arcpy.GetParameterAsText(1)
predict_rasters = ConversionUtils.SplitMultiInputs(predict_raster_files)
#default_nodata_value = -3.4028234663852886E+38

def get_feature_mask(inputFeature,predict_raster):
    sampleRaster = arcpy.Raster(predict_raster)
    arcpy.env.extent = sampleRaster.extent
    leftlow_point = arcpy.Point(sampleRaster.extent.XMin,sampleRaster.extent.YMin)
    arcpy.env.snapRaster = predict_raster
    arcpy.env.cellSize = sampleRaster
    metaRaster = sampleRaster*0 + 1
    sampleRasterArray = arcpy.RasterToNumPyArray(sampleRaster,lower_left_corner = leftlow_point,ncols = metaRaster.width, nrows = metaRaster.height, nodata_to_value = default_nodata_value)
    exist_value_mask  = numpy.where(sampleRasterArray == default_nodata_value,0,1).astype("bool")
    arcpy.MakeFeatureLayer_management(inputFeature, "lyr")
    feature_raster = arcpy.sa.ExtractByMask(metaRaster, "lyr")
    feature_mask = arcpy.RasterToNumPyArray(feature_raster,nodata_to_value = 0).astype("bool")
    arcpy.Delete_management("lyr")
    return exist_value_mask&feature_mask

def get_exist_value_mask(sample_raster):
    sample_2d_array = arcpy.RasterToNumPyArray(sample_raster,nodata_to_value = default_nodata_value)
    exist_value_mask  = numpy.where(sample_2d_array == default_nodata_value,0,1).astype("bool")
    return exist_value_mask

def get_nodata_value(sample_raster):
    sampleRaster = arcpy.Raster(sample_raster)
    nodata_value = sampleRaster.noDataValue
    return nodata_value

def get_fpr_tpr_auc(feature_mask,exist_value_mask,predict_raster):
    predict_2d_array = arcpy.RasterToNumPyArray(predict_raster,nodata_to_value = default_nodata_value)
    feature_1d_array = feature_mask[exist_value_mask]
    predict_1d_array = predict_2d_array[exist_value_mask]
    fpr, tpr, thresholds = roc_curve(feature_1d_array, predict_1d_array)
    roc_auc = auc(fpr, tpr)
    if roc_auc  < 0.5:
        predict_1d_array = - predict_1d_array
    fpr, tpr, thresholds = roc_curve(feature_1d_array, predict_1d_array)
    roc_auc = auc(fpr, tpr)
    raster_name = os.path.basename(os.path.splitext(predict_raster)[0])
    return fpr, tpr, roc_auc, raster_name

def draw_rocs(inputFeature,predict_rasters):
    feature_mask = get_feature_mask(inputFeature,predict_rasters[0])
    exist_value_mask = get_exist_value_mask(predict_rasters[0])
    fpr_tpr_auc_list = []
    for predict_raster in predict_rasters:
        fpr_tpr_auc_list.append(get_fpr_tpr_auc(feature_mask,exist_value_mask,predict_raster))
    fig = pylab.figure()
    ax = fig.add_subplot(1,1,1)
    for fpr_tpr_auc in fpr_tpr_auc_list:
        ax.plot(fpr_tpr_auc[0], fpr_tpr_auc[1], label='%s (AUC = %0.3f)' %(fpr_tpr_auc[3],fpr_tpr_auc[2]))
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_aspect("equal")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
    fig.show()

default_nodata_value = get_nodata_value(predict_rasters[0])
draw_rocs(inputFeature,predict_rasters)
arcpy.CheckInExtension("spatial")