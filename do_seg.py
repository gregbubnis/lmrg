import glob
import json
import os
import napari
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import match_template, peak_local_max
from skimage.segmentation import watershed
from skimage import measure
from skimage import filters
from skimage import morphology
import tifffile as tf

def make_gauss_template(sig):
    """make 3D gaussian template from a vector of sigmas"""
    vecs = []
    for s in sig:
        hw = np.ceil(s*3)
        ix = np.arange(-hw, hw+1)            
        x = np.exp(-0.5*(ix*ix)/s**2)
        vecs.append(x)
        #print(s, hw, ix, x)
    tt = np.multiply.outer(vecs[0], vecs[1])
    tt = np.multiply.outer(tt, vecs[2])
    #tt /= np.sum(tt.ravel())
    return tt

def make_pb_template(sig):
    """make 3D pillbox (hard edge ellipse) template from a vector of sigmas"""
    vecs = []
    for s in sig:
        hw = np.ceil(s*1.1)
        ix = np.arange(-hw, hw+1)            
        x = np.exp(-0.5*(ix*ix)/s**2)
        vecs.append(x)
        #print(s, hw, ix, x)
    tt = np.multiply.outer(np.multiply.outer(vecs[0], vecs[1]), vecs[2])
    tt = np.array(tt>=np.exp(-0.5))
    return tt

def get_props(x, labels, scale):
    """extract properties from a segmented volume
    
    x: (ndarray) raw volume
    labels: (ndarray) ROI int label mask w same dims as x
    scale: (list) 3-vector for voxel size in um
    """
    cell_ix = np.where(labels !=0)
    lvec = labels[cell_ix]
    unique_labels = np.sort(np.unique(lvec))        
    data = []
    for i in unique_labels:
        lix = np.where(lvec==i)[0]
        voxels = (cell_ix[0][lix], cell_ix[1][lix], cell_ix[2][lix])
        zyx_ics = np.mean(voxels, axis=1)
        zyx_lcs = zyx_ics*np.asarray(scale)
        intensity = x[voxels]
        dd = dict(
            z=zyx_lcs[0],
            y=zyx_lcs[1],
            x=zyx_lcs[2],
            intensity=np.sum(intensity),
            volume=len(lix)*np.product(scale)
            )
        data.append(dd)
    return pd.DataFrame(data)

def do_seg_fish(f, params, view=False, dest='results'):
    """segment and compute properties for FISH ROIs

    Parameters:
    -----------
    f: (str) tiff filename
    params: (dict) extraction parameters
    view: (bool) launch napari viewer to inspect
    dest: (str) destination folder

    Returns:
    --------
    df_props: (pandas DataFrame) dataframe of cell properties (pos/size)

    Exports ```df_props``` and ```params``` to csv and json, respectively.

    cooc8upfasegG
    Uses a template filter + thresholding to generate an ROI mask
    """
    myname = params['myname']
    template_std = params['template_std']
    template_threshold = params['template_threshold']
    scale = params['voxel_size']

    x = tf.TiffFile(f).asarray()
    os.makedirs(dest, exist_ok=True)
    print("--------"*7)
    print('%-20s:' % 'file' , f )
    for k, v in params.items():
        print('%-20s:' %k , v )

    ## template filter
    template = make_gauss_template(template_std)
    res = match_template(x, template, pad_input=True)
    filtered = res*np.array(res>template_threshold)
    labels, _ = ndi.label(filtered)
    
    if view:
        v = napari.Viewer()
        v.add_image(x, opacity=0.5, scale=scale, name='raw', blending='additive', interpolation='nearest')
        v.add_image(template, scale=scale, name='template', blending='additive', interpolation='nearest', contrast_limits=[0, 0.1])
        v.add_labels(labels, scale=scale, name='labels', blending='additive')
        napari.run()

    ## extract properties and export results
    df_props = get_props(x, labels, scale)
    df_props.to_csv(os.path.join(dest, '%s_%s.csv' % (myname, os.path.basename(f))), float_format='%6g')
    with open(os.path.join(dest, '%s_%s.json' % (myname, os.path.basename(f))), 'w') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    print(df_props)
    return df_props

def do_seg_nuclei(f, params, view=False, dest='results'):
    """segment and compute properties for close packed nuceli

    Parameters:
    -----------
    f: (str) tiff filename
    params: (dict) extraction parameters
    view: (bool) launch napari viewer to inspect
    dest: (str) destination folder

    Returns:
    --------
    df_props: (pandas DataFrame) dataframe of cell properties (pos/size)

    Exports ```df_props``` and ```params``` to csv and json, respectively.

    Workflow:
    1. Compute local otsu threshold and use to segment a foreground mask
    2. Use a pillbox template filter to extract disconnected central blobs, one per cell
    3. Watershed segmentation using the mask (1.) and centers (2.)
    """
    myname = params['myname']
    template_std = np.asarray(params['template_std'])
    template_threshold = params['template_threshold']
    local_otsu_kernel = np.asarray(params['local_otsu_kernel'])
    scale = params['voxel_size']

    x = tf.TiffFile(f).asarray()
    os.makedirs(dest, exist_ok=True)
    print("--------"*7)
    print('%-20s:' % 'file' , f )
    for k, v in params.items():
        print('%-20s:' %k , v )

    ## threshold masks
    li_mask = x > filters.threshold_li(x)
    otsu_mask = x > filters.threshold_otsu(x)
    otsu_masked = x*otsu_mask
    template = make_pb_template(local_otsu_kernel)
    local_otsu = (x>filters.rank.otsu(x, template)) * otsu_mask
    local_otsu_eroded = morphology.erosion(local_otsu, morphology.ball(3))
    local_otsu_dilated = morphology.dilation(local_otsu_eroded, morphology.ball(3))

    ## isolate the outer "shell" (not used currently)
    # denoised = ndi.median_filter(x, size=5)
    # smoothed = filters.gaussian(denoised, sigma=template_std*0.05)
    # edges2 = filters.scharr(smoothed)
    # otsu_scharr = edges2*(edges2 > filters.threshold_otsu(edges2))

    ## pillbox template filter to get centers (blobs too small to be good masks)
    template = make_pb_template(template_std)
    res = match_template(otsu_masked, template, pad_input=True)
    filtered = res*np.array(res>template_threshold)
    labels = measure.label(filtered>0)
    props = measure.regionprops(labels)
    peaks = np.rint(np.asarray([x.centroid for x in props])).astype(int)

    ## watershed
    mask = np.zeros(x.shape, dtype=bool)
    mask[tuple(peaks.T)] = True
    markers, _ = ndi.label(mask)
    markers = morphology.dilation(markers, morphology.ball(5))
    for_ws = -1*filters.gaussian(np.asarray(markers>0, dtype=int), sigma=template_std*0.5)
    ws_labels = watershed(for_ws, markers=markers, mask=local_otsu_dilated>0)

    if view:
        v = napari.Viewer()
        v.add_labels(ws_labels, scale=scale, name='ws_labels', blending='additive')
        v.add_image(-for_ws, visible=False, opacity=0.5, scale=scale, colormap='blue', name='for_ws', blending='additive', interpolation='nearest')
        v.add_image(local_otsu_dilated, visible=False, opacity=0.5, scale=scale, colormap='blue', name='local_otsu_dilated', blending='additive', interpolation='nearest')
        v.add_image(local_otsu_eroded, visible=False, opacity=0.5, scale=scale, colormap='blue', name='local_otsu_eroded', blending='additive', interpolation='nearest')
        v.add_image(local_otsu, visible=False, opacity=0.5, scale=scale, colormap='blue', name='local_otsu', blending='additive', interpolation='nearest')
        v.add_image(li_mask, visible=False, opacity=0.5, scale=scale, colormap='green', name='li_mask', blending='additive', interpolation='nearest', contrast_limits=[0, 1])
        v.add_image(otsu_mask, visible=False, opacity=0.5, scale=scale, colormap='blue', name='otsu_mask', blending='additive', interpolation='nearest', contrast_limits=[0, 1])
        v.add_image(otsu_masked, visible=False, opacity=0.5, scale=scale, colormap='blue', name='otsu_masked', blending='additive', interpolation='nearest')
        v.add_points(peaks, scale=scale, size=5)
        v.add_labels(labels, visible=False, scale=scale, name='labels', blending='additive')
        v.add_image(template, visible=False, scale=scale, name='template', blending='additive', interpolation='nearest', contrast_limits=[0, 0.1])
        v.add_image(filtered, visible=False, opacity=0.5, scale=scale, colormap='magenta', name='template_filtered', blending='additive', interpolation='nearest', contrast_limits=[0, 1])
        v.add_image(x, opacity=0.5, scale=scale, name='raw', blending='additive', interpolation='nearest')
        napari.run()

    ## extract properties and export results
    df_props = get_props(x, ws_labels, scale)
    df_props.to_csv(os.path.join(dest, '%s_%s.csv' % (myname, os.path.basename(f))), float_format='%6g')
    with open(os.path.join(dest, '%s_%s.json' % (myname, os.path.basename(f))), 'w') as f:
        json.dump(params, f, indent=2)
        f.write('\n')
    print(df_props)
    return df_props


if __name__ == "__main__":

    ## FISH
    view = False
    flz = sorted(glob.glob('data/fish*tiff'))
    dest = 'results'
    params = dict(
        myname='Bubnis_Greg',
        voxel_size=[0.200, 0.162, 0.162],
        template_std=[0.5, 0.7, 0.7],
        template_threshold=0.35,
    )
    do_seg_fish(flz[0], view=view, params=params, dest='results')
    do_seg_fish(flz[1], view=view, params=params, dest='results')
    do_seg_fish(flz[2], view=view, params=params, dest='results')
    do_seg_fish(flz[3], view=view, params=params, dest='results')

    # NUCLEI
    view = False
    flz = sorted(glob.glob('data/nuclei*tif'))
    params_1 = dict(
        myname='Bubnis_Greg',
        voxel_size=[0.200, 0.124, 0.124],
        template_std=[16, 24, 24],
        local_otsu_kernel=[2, 30, 30],
        template_threshold=0.4,
    )
    params_4 = dict(
        myname='Bubnis_Greg',
        voxel_size=[0.200, 0.124, 0.124],
        template_std=[16, 24, 24],
        local_otsu_kernel=[2, 30, 30],
        template_threshold=0.3,
    )
    do_seg_nuclei(flz[0], view=view, params=params_1, dest='results')
    do_seg_nuclei(flz[1], view=view, params=params_1, dest='results')
    do_seg_nuclei(flz[2], view=view, params=params_1, dest='results')
    do_seg_nuclei(flz[3], view=view, params=params_4, dest='results')

    # CALIBRATIONS
    params_cal = dict(
        myname='Bubnis_Greg_calibration_fish',
        voxel_size=[1, 1, 1],
        template_std=[3, 3, 3],
        template_threshold=0.3,
    )
    do_seg_fish('data/calibration.tiff', view=view, params=params_cal, dest='results')

    params_cal = dict(
        myname='Bubnis_Greg_calibration_nuclei',
        voxel_size=[1, 1, 1],
        template_std=[7, 7, 7],
        local_otsu_kernel=[2, 5, 5],
        template_threshold=0.3,
    )
    do_seg_nuclei('data/calibration.tiff', view=view, params=params_cal, dest='results')

