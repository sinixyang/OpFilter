# -*- coding: utf-8 -*-
"""
Created on Fri Mar 06 04:46:02 2015

@author: Sinix-Studio
"""
import arcpy
import numpy
import os
import ConversionUtils
from sklearn.metrics import roc_curve, auc
from chaco.api import create_line_plot, OverlayPlotContainer, VPlotContainer,LinePlot
from chaco.tools.api import PanTool, ZoomTool
from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from enable.component_editor import ComponentEditor
from traits.api import HasTraits, Instance,Button,CArray,CInt,CStr,CFloat,Enum
from traitsui.api import View, Item,  VGroup,HGroup


def get_low_pass_circle_mask(meta_mask,radius = 1,mode = 0, order = 2):
    mask = numpy.zeros_like(meta_mask,dtype = "float")
    row_center = mask.shape[0]/2.0
    col_center = mask.shape[1]/2.0
    distance = lambda x,y,xc,yc:((x - xc)**2 + (y - yc)**2)**0.5
    if mode == 0:#ideal
        for row_id in range(0,mask.shape[0]):
            for col_id in range(0,mask.shape[1]):
                if distance(row_id,col_id,row_center,col_center) <= radius:
                    mask[row_id,col_id] = 1
    elif mode == 1:#bufferworth
        for row_id in range(0,mask.shape[0]):
            for col_id in range(0,mask.shape[1]):
                mask[row_id,col_id] =  1/(1 + (distance(row_id,col_id,row_center,col_center)/radius)**(order*2))
    elif mode == 2:#gauss
        for row_id in range(0,mask.shape[0]):
            for col_id in range(0,mask.shape[1]):
                mask[row_id,col_id] =  numpy.exp(-0.5*(distance(row_id,col_id,row_center,col_center)/radius)**2)
    return mask

def get_high_pass_circle_mask(meta_mask,radius = 1,mode = 0, order = 2):
    mask = numpy.ones_like(meta_mask,dtype = "float")
    #print meta_mask.shape
    row_center = mask.shape[0]/2.0
    col_center = mask.shape[1]/2.0
    distance = lambda x,y,xc,yc:((x - xc)**2 + (y - yc)**2)**0.5
    if mode == 0:#ideal
        for row_id in range(0,mask.shape[0]):
            for col_id in range(0,mask.shape[1]):
                if distance(row_id,col_id,row_center,col_center) <= radius:
                    mask[row_id,col_id] = 0
    elif mode == 1:#bufferworth
        for row_id in range(0,mask.shape[0]):
            for col_id in range(0,mask.shape[1]):
                mask[row_id,col_id] =  1 - 1/(1 + (distance(row_id,col_id,row_center,col_center)/radius)**(order*2))
    elif mode == 2:#gauss
        for row_id in range(0,mask.shape[0]):
            for col_id in range(0,mask.shape[1]):
                mask[row_id,col_id] =  1 - numpy.exp(-0.5*(distance(row_id,col_id,row_center,col_center)/radius)**2)
    return mask


def get_fisher_measure(predict_2d_array,common_mask,feature_mask):
    common_mean = predict_2d_array[common_mask].mean()
    common_std = predict_2d_array[common_mask].std()
    deposit_mean = predict_2d_array[common_mask&feature_mask].mean()
    deposit_std = predict_2d_array[common_mask&feature_mask].std()
    return ((common_mean - deposit_mean)**2/(common_std**2 + deposit_std**2))

def get_auc(predict_2d_array,common_mask,feature_mask):
    feature_1d_array = feature_mask[common_mask]
    predict_1d_array = predict_2d_array[common_mask]
    fpr, tpr, thresholds = roc_curve(feature_1d_array, predict_1d_array)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        predict_1d_array = - predict_1d_array
        fpr, tpr, thresholds = roc_curve(feature_1d_array, predict_1d_array)
        roc_auc = auc(fpr, tpr)
        return roc_auc
    else:
        return roc_auc

def mapping_radius_score(img_ext,common_mask,feature_mask,object_func = 0,filter_func = 0,func_order = 2, max_radius = 0):
    img_fft = numpy.fft.fft2(img_ext)
    if max_radius == 0:
        max_radius = int(numpy.array(img_fft.shape).min()/4)# to reduce computational
    else:
        pass
    sig_low_pass_list = []
    sig_high_pass_list = []
    high_pass_score = filter_score(img_ext,common_mask,feature_mask,object_func,1,filter_func,func_order)
    low_pass_score = filter_score(img_ext,common_mask,feature_mask,object_func,0,filter_func,func_order)
    for radius in range(1,max_radius + 1):
        sig_low_pass_list.append(low_pass_score(radius))
        sig_high_pass_list.append(high_pass_score(radius))
    sig_low_pass = numpy.array(sig_low_pass_list)
    sig_high_pass = numpy.array(sig_high_pass_list)
    return sig_low_pass,sig_high_pass

def get_extend_2d_array(in_img,new_shape,fill_value = 0):
    new_2d_array = numpy.zeros(new_shape,dtype = in_img.dtype)
    new_2d_array.fill(fill_value)
    in_img_shape = in_img.shape
    row_extend_upper_origin = int((new_shape[0] - in_img_shape[0])/2)
    col_extend_left_origin = int((new_shape[1] - in_img_shape[1])/2)
    new_2d_array[row_extend_upper_origin:row_extend_upper_origin + in_img_shape[0],col_extend_left_origin : col_extend_left_origin + in_img_shape[1]] = in_img
    return new_2d_array

def get_equal_weight(distance):#also could construct other weight function for diffusion model
    return 1

def get_weight_window(radius,weight_func):
    weight_window = numpy.zeros((2*radius + 1,2*radius + 1),dtype = float)
    for row_id in range(0,2*radius + 1):
        for col_id in range(0,2*radius + 1):
            distance = ((row_id - radius)**2 + (col_id - radius)**2)**0.5
            weight_window[row_id,col_id] = weight_func(distance)
    weight_window = weight_window/weight_window.sum()
    return weight_window


def extrapolation(ultra_extend_img,ultra_extend_extra_mask,weight_window):
    ultra_extend_shape = ultra_extend_extra_mask.shape
    temp_ultra_extend_img = ultra_extend_img.copy()
    weight_window_radius = int((weight_window.shape[0] - 1)/2)
    for row_id in range(0,ultra_extend_shape[0]):
        for col_id in range(0,ultra_extend_shape[1]):
            if ultra_extend_extra_mask[row_id,col_id] == 1:
                window_data = ultra_extend_img[row_id - weight_window_radius:row_id + weight_window_radius +1,col_id - weight_window_radius:col_id + weight_window_radius +1]
                temp_ultra_extend_img[row_id,col_id] = (window_data*weight_window).sum()
    return temp_ultra_extend_img

def get_extrapolated_2d_array(in_img,in_common_mask,new_shape,maximum_iter = 20,weight_window_radius = 1,weight_func = get_equal_weight):
    img = in_img.copy()
    common_mask = in_common_mask.copy()
    ulra_extend_shape = (new_shape[0]+ 2*weight_window_radius,new_shape[1]+ 2*weight_window_radius)
    ultra_extend_img = get_extend_2d_array(img,ulra_extend_shape)
    extend_mask = get_extend_2d_array(common_mask,new_shape)
    ultra_extend_mask = get_extend_2d_array(extend_mask,ulra_extend_shape,fill_value = 0)
    ultra_extend_extra_mask = 1 - get_extend_2d_array(extend_mask,ulra_extend_shape,fill_value = 1)
    temp_ultra_extend_img = ultra_extend_img.copy()
    temp_ultra_extend_img[~ultra_extend_mask] = temp_ultra_extend_img[ultra_extend_mask].mean()
    weight_window = get_weight_window(weight_window_radius,weight_func)
    for iter_id in range(0,maximum_iter):
        temp_ultra_extend_img = extrapolation(temp_ultra_extend_img,ultra_extend_extra_mask,weight_window)
    extend_img = temp_ultra_extend_img[weight_window_radius:weight_window_radius + new_shape[0],weight_window_radius:weight_window_radius + new_shape[1]]
    return extend_img

def get_low_pass_img(in_img,raw_shape,threshold,filter_func = 0, order = 2):#in_img is an extend img
    extend_shape = in_img.shape
    row_extend_upper_origin = int((extend_shape[0] - raw_shape[0])/2)
    col_extend_left_origin = int((extend_shape[1] - raw_shape[1])/2)
    img = in_img.copy()
    img_fft = numpy.fft.fft2(img)
    low_pass_mask = get_low_pass_circle_mask(img_fft,threshold,mode = filter_func, order = order)
    low_pass_mask = numpy.fft.fftshift(low_pass_mask)
    temp_low_pass_img = numpy.fft.ifft2(img_fft*low_pass_mask).real
    low_pass_img = temp_low_pass_img[row_extend_upper_origin:row_extend_upper_origin + raw_shape[0],col_extend_left_origin : col_extend_left_origin + raw_shape[1]]
    return low_pass_img

def get_high_pass_img(in_img,raw_shape,threshold,filter_func = 0, order = 2):
    extend_shape = in_img.shape
    #print extend_shape
    row_extend_upper_origin = int((extend_shape[0] - raw_shape[0])/2)
    col_extend_left_origin = int((extend_shape[1] - raw_shape[1])/2)
    img = in_img.copy()
    img_fft = numpy.fft.fft2(img)
    high_pass_mask = get_high_pass_circle_mask(img_fft,threshold,mode = filter_func, order = order)
    high_pass_mask = numpy.fft.fftshift(high_pass_mask)
    temp_high_pass_img = numpy.fft.ifft2(img_fft*high_pass_mask).real
    high_pass_img = temp_high_pass_img[row_extend_upper_origin:row_extend_upper_origin + raw_shape[0],col_extend_left_origin : col_extend_left_origin + raw_shape[1]]
    return high_pass_img

class filter_score:
    def __init__(self,img_ext,common_mask,feature_mask,object_func = 0,filter_mode = 0,filter_func = 2,func_order = 2):
        self.img_ext = img_ext
        self.raw_shape = common_mask.shape
        self.filter_func = filter_func
        self.func_order = func_order
        self.common_mask = common_mask
        self.feature_mask = feature_mask
        self.extend_shape = img_ext.shape
        self.raw_shape = common_mask.shape
        self.row_origin = int((self.extend_shape[0] - self.raw_shape[0])/2)
        self.col_origin = int((self.extend_shape[1] - self.raw_shape[1])/2)
        self.row_end = self.row_origin + self.raw_shape[0]
        self.col_end = self.col_origin + self.raw_shape[1]
        self.img_ext_fd = numpy.fft.fft2(img_ext)
        if object_func == 0:
            self.object_func = get_fisher_measure
        elif object_func == 1:
            self.object_func = get_auc
        else:
            pass
        if filter_mode == 0:
            self.get_filter_mask = get_low_pass_circle_mask
        elif filter_mode == 1:
            self.get_filter_mask = get_high_pass_circle_mask
        else:
            pass

    def __call__(self,radius):
        filter_mask = self.get_filter_mask(self.img_ext,radius,self.filter_func, self.func_order)
        filter_mask = numpy.fft.fftshift(filter_mask)
        ext_filtered_img =  numpy.fft.ifft2(self.img_ext_fd*filter_mask).real
        filtered_img = ext_filtered_img[self.row_origin:self.row_end,self.col_origin:self.col_end]
        measure = self.object_func(filtered_img,self.common_mask,self.feature_mask)
        return measure

def main():
    predictive_feature_file = arcpy.GetParameterAsText(0)
    input_raster_files = arcpy.GetParameterAsText(1)
    measure = arcpy.GetParameterAsText(2)
    filter_functions = arcpy.GetParameterAsText(3)
    butterworth_order = arcpy.GetParameterAsText(4)
    max_wavenumber = arcpy.GetParameterAsText(5)
    saving_folder = arcpy.GetParameterAsText(6)
    arcpy.CheckOutExtension("Spatial")
    butterworth_order = int(butterworth_order)
    func_list =ConversionUtils.SplitMultiInputs(filter_functions)
    max_wavenumber = int(max_wavenumber)
    
    input_raster_file_list = ConversionUtils.SplitMultiInputs(input_raster_files)
    sampleRaster = arcpy.Raster(input_raster_file_list[0])
    arcpy.env.extent = sampleRaster.extent
    arcpy.env.snapRaster = sampleRaster
    leftlow_point = arcpy.Point(sampleRaster.extent.XMin,sampleRaster.extent.YMin)
    ori_ycellsize = sampleRaster.meanCellHeight
    ori_xcellsize = sampleRaster.meanCellWidth
    nodata_value = sampleRaster.noDataValue
    prj = sampleRaster.spatialReference.exporttostring()
    prj_exist = True
    if sampleRaster.spatialReference.name == u'Unknown':
        prj_exist = False
    else:
        pass
    metaRaster = sampleRaster*0 + 1
    sample_raster_array = arcpy.RasterToNumPyArray(sampleRaster,lower_left_corner = leftlow_point,ncols = metaRaster.width, nrows = metaRaster.height, nodata_to_value = nodata_value)
    feature_mask_raster = arcpy.sa.ExtractByMask(metaRaster, predictive_feature_file)
    
    #arcpy.AddMessage ("sample raster shape" + str(sample_raster_array.shape[0])+str(sample_raster_array.shape[1]))
    feature_mask = arcpy.RasterToNumPyArray(feature_mask_raster,nodata_to_value = 0).astype("bool")
    common_mask = numpy.where(sample_raster_array == nodata_value,0,1).astype("bool")
    img_ori_shape = common_mask.shape
    img_ori_dict = {}
    for raster_file in input_raster_file_list:
        img_ori_dict[os.path.split(raster_file)[-1]] = arcpy.RasterToNumPyArray(raster_file,nodata_to_value = 0)
    
    power =  int(numpy.log2(numpy.array(common_mask.shape).max())) + 1
    img_ext_shape = (2**power,2**power)
    scale = (2**power)*ori_ycellsize
    if measure == "AUC":
        measure_code = 1
    else:
        measure_code = 0
    
    img_func_score_dict = {}
    img_ext_dict = {}
    for img_name in img_ori_dict:
        extend_interpolated_img = get_extrapolated_2d_array(img_ori_dict[img_name],common_mask,img_ext_shape,maximum_iter = 0)
        img_ext_dict[img_name] = extend_interpolated_img
        func_score_dict = {}
        for func in func_list:
            if func == "Ideal":
                func_code = 0
            elif func == "Butterworth":
                func_code = 1
            elif func == "Gauss": 
                func_code = 2
            score_dict = {}
            score_dict["Low Pass"],score_dict["High Pass"] = mapping_radius_score(extend_interpolated_img,common_mask,feature_mask,object_func = measure_code,filter_func = func_code,func_order = butterworth_order,max_radius = max_wavenumber)
            func_score_dict[func] = score_dict
        img_func_score_dict[img_name] = func_score_dict
    
    filtering_result_dict = {}
    class Opfilter_GUI(HasTraits):
        plot = Instance(VPlotContainer)
        cursor = Instance(BaseCursorTool)
        #cursor_pos = DelegatesTo('cursor', prefix='current_position')
        Wavenumber = CInt()
        Measure = CFloat()
        Wavelength = CInt()
        threshold = CInt()
        saving_name = CStr()
        img_names = tuple(img_func_score_dict.keys())
        Raster = Enum(img_names)
        Mode = Enum("High Pass","Low Pass")
        Function = Enum(tuple(func_list))
        save_button = Button("Save")
        line = Instance(LinePlot)
        init_plot_array = numpy.array([0,5,10],dtype = "float")
        x = CArray(value =init_plot_array, dtype = "float")
        y = CArray(value =init_plot_array, dtype = "float")
    
        def __init__(self):
            super(Opfilter_GUI, self).__init__()
            container = VPlotContainer(padding=0, spacing=20)
            self.plot = container
            subcontainer = OverlayPlotContainer(padding=40)
            container.add(subcontainer)
    
            self.line = create_line_plot([self.x, self.y], add_grid=True,
                                    add_axis=True, index_sort='ascending',
                                    orientation = 'h')
            subcontainer.add(self.line)
            csr = CursorTool(self.line,
                            drag_button="left",
                            color='blue')
            self.cursor = csr
            self.threshold = csr.current_position[0]
            csr.current_position = 0.5, 0.5
            csr.on_trait_change(self._cursor_changed,"current_position")
            self.line.overlays.append(csr)
            self.line.tools.append(PanTool(self.line, drag_button="middle",restrict_to_data = True))
            self.line.overlays.append(ZoomTool(self.line))
            self.on_trait_change(self._save_filtering_result,"save_button")
            self.on_trait_change(self._gen_saving_name,"Raster")
            self.on_trait_change(self._gen_saving_name,"Function")
            self.on_trait_change(self._gen_saving_name,"Mode")
            self.on_trait_change(self._re_graph,"Raster")
            self.on_trait_change(self._re_graph,"Function")
            self.on_trait_change(self._re_graph,"Mode")
            self.on_trait_change(self._check_saving_name,"saving_name")
    
        def _gen_saving_name(self):
            if self.Mode == "High Pass":
                midle_name = "hp"
            elif self.Mode == "Low Pass":
                midle_name = "lp"
            if self.Function == "Ideal":
                last_name = "id"
            elif self.Function == "Butterworth":
                last_name = "bw"
            elif self.Function == "Gauss":
                last_name = "gs"
            self.saving_name = self.Raster + midle_name + last_name
        
        def _check_saving_name(self):
            if len(self.saving_name) > 13:
                self.saving_name = self.saving_name[0:13]
        
        def _cursor_changed(self):
            self.threshold = self.cursor.current_position[0]
            self.Wavenumber = self.cursor.current_position[0]
            self.Measure = self.cursor.current_position[1]
            self.Wavelength = scale/self.cursor.current_position[0]
    
        def _re_graph(self):
            y = img_func_score_dict[self.Raster][self.Function][self.Mode]
            x = numpy.arange(1,1 + y.shape[0],dtype = "int")
            self.x = x
            self.y = y
            self.__init__()
    
        def _save_filtering_result(self):
            if self.Mode == "High Pass":
                get_filtering_img = get_high_pass_img
            elif self.Mode == "Low Pass":
                get_filtering_img = get_low_pass_img
            else:
                pass
            if self.Function == "Ideal":
                self.filter_func = 0
            elif self.Function == "Butterworth":
                self.filter_func = 1
            elif self.Function == "Gauss":
                self.filter_func = 2
            else:
                pass
            img_ext = img_ext_dict[self.Raster]
            print "saving",Item('Measure',style = "readonly",label = measure)
            filtering_result_dict[self.saving_name] = get_filtering_img(img_ext,img_ori_shape,self.threshold,self.filter_func,butterworth_order)
    
        traits_view = View(VGroup(Item('plot',editor=ComponentEditor(),resizable=True, springy=True,show_label=False),
                                  HGroup(Item('Wavenumber',style = "readonly"),Item('Wavelength',style = "readonly", width=10)),
                                  Item('Measure',style = "readonly",label = measure),
                                  HGroup(Item("Raster"),Item("Function"),Item("Mode"),Item("threshold", width=3),Item("saving_name",width=13,label = "Raster Name"),Item("save_button",show_label=False))),
                            title="Optimal Filter",
                            resizable=True,
                            width=850,
                            height=500)
    gui = Opfilter_GUI()
    gui.configure_traits()
    
    for result_name in filtering_result_dict:
        result_img = filtering_result_dict[result_name]
        result_img[~common_mask] = nodata_value
        tempRaster = arcpy.NumPyArrayToRaster(result_img,leftlow_point,ori_ycellsize,ori_xcellsize,nodata_value)
        saving_route = saving_folder + "//" + result_name
        tempRaster.save(saving_route)
        if prj_exist == False:
            pass
        else:
            arcpy.DefineProjection_management(saving_route, prj)
    arcpy.CheckInExtension("Spatial")

if __name__ == '__main__':
    main()