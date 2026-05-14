import os as _os
_vtp_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    'bilayer_sim0001.vtp'
)

# state file generated using paraview version 6.1.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 1

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[1561, 537],
    CenterOfRotation=[249.61142349243164, 250.3909034729004, -0.2740001678466797],
    CameraPosition=[247.04269848799944, 901.252210160883, 162.94797928816465],
    CameraFocalPoint=[249.6114234924316, 250.39090347290036, -0.2740001678466909],
    CameraViewUp=[0.0016807366477598058, -0.2432396143889024, 0.9699647751935423],
)

SetActiveView(None)

layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1561, 537)

SetActiveView(renderView1)

selection_sources23004 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.23004', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)&(leaflet == 0)',
    Assembly='',
    Selectors=['/'])

selection_sources26087 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.26087', groupname='selection_sources', ElementType='Point Data',
    QueryString='(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter23037 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.23037', groupname='selection_sources', Input=selection_sources23004,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources23049 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.23049', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)&(leaflet == 1)',
    Assembly='',
    Selectors=['/'])

selection_sources24680 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24680', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 1)&(leaflet == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24713 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24713', groupname='selection_sources', Input=selection_sources24680,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter23082 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.23082', groupname='selection_sources', Input=selection_sources23049,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter26120 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.26120', groupname='selection_sources', Input=selection_sources26087,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24479 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24479', groupname='selection_sources', ElementType='Point Data',
    QueryString='(in_raft == 1)&(leaflet == 1)&(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24512 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24512', groupname='selection_sources', Input=selection_sources24479,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24591 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24591', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 1)&(leaflet == 0)',
    Assembly='',
    Selectors=['/'])

selection_filter24624 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24624', groupname='selection_sources', Input=selection_sources24591,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24524 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24524', groupname='selection_sources', ElementType='Point Data',
    QueryString='(in_raft == 1)&(leaflet == 0)&(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24557 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24557', groupname='selection_sources', Input=selection_sources24524,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources25799 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.25799', groupname='selection_sources', ElementType='Point Data',
    QueryString='(pip_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter25832 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.25832', groupname='selection_sources', Input=selection_sources25799,
    Expression='s0',
    SelectionNames=['s0'])

selectionSource0 = CreateSelection(proxyname='SelectionQuerySource', registrationname='SelectionSource0', groupname='selections', ElementType='Point Data',
    QueryString='(is_head == 1)',
    Assembly='',
    Selectors=['/'])

appendSelections = CreateSelection(proxyname='AppendSelections', registrationname='AppendSelections', groupname='selections', Input=selectionSource0,
    Expression='s0',
    SelectionNames=['s0'])

tails_Source = XMLPolyDataReader(registrationName='Tails_Source', FileName=[_vtp_path])
tails_Source.Set(
    PointArrayStatus=['region', 'in_raft', 'electron_density', 'phase'],
    TimeArray='None',
)

general = XMLPolyDataReader(registrationName='General', FileName=[_vtp_path])
general.Set(
    PointArrayStatus=['region', 'is_head', 'is_glycerol', 'is_tail', 'is_protein', 'order_param', 'in_raft', 'is_pip', 'pip_head', 'electron_density', 'lipid_id', 'leaflet', 'bead_type', 'seg_idx', 'n_doublebonds', 'phase', 'chain_length'],
    TimeArray='None',
)

head_out = ExtractSelection(registrationName='Head_out', Input=general,
    Selection=selection_filter23037)

glyc_out = ExtractSelection(registrationName='Glyc_out', Input=general,
    Selection=selection_filter24713)

proteins = Threshold(registrationName='Proteins', Input=general)
proteins.Set(
    Scalars=['POINTS', 'is_protein'],
    LowerThreshold=1.0,
    UpperThreshold=1.0,
)

raft_in = ExtractSelection(registrationName='Raft_in', Input=general,
    Selection=selection_filter24512)

cHOL = Threshold(registrationName='CHOL', Input=general)
cHOL.Set(
    Scalars=['POINTS', 'region'],
    LowerThreshold=4.0,
    UpperThreshold=4.0,
)

head_in = ExtractSelection(registrationName='Head_in', Input=general,
    Selection=selection_filter23082)

glyc_in = ExtractSelection(registrationName='Glyc_in', Input=general,
    Selection=selection_filter24624)

raft_out = ExtractSelection(registrationName='Raft_out', Input=general,
    Selection=selection_filter24557)

tails = Threshold(registrationName='Tails', Input=tails_Source)
tails.Set(
    Scalars=['POINTS', 'region'],
    LowerThreshold=2.0,
    UpperThreshold=3.0,
)

e_density = ExtractSelection(registrationName='E_density', Input=general,
    Selection=selection_filter26120)

pIPs = ExtractSelection(registrationName='PIPs', Input=head_in,
    Selection=selection_filter25832)

tails_Poly = ExtractSurface(registrationName='Tails_Poly', Input=tails)

tails_Tube = Tube(registrationName='Tails_Tube', Input=tails_Poly)
tails_Tube.Set(
    Scalars=['POINTS', ''],
    Vectors=['POINTS', ''],
    NumberofSides=12,
    Radius=1.2,
)

appendSelections.SetSelectionId(general.GetGlobalID())
appendSelections.SetSelectionPort(0)

proteinsDisplay = Show(proteins, renderView1, 'UnstructuredGridRepresentation')
proteinsDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.753, 0.753, 0.753],
    ColorArrayName=['FIELD', ''],
    DiffuseColor=[0.753, 0.753, 0.753],
    MapScalars=0,
    Opacity=0.5,
    GaussianRadius=2.5,
)
proteinsDisplay.ScaleTransferFunction.Points = [9.0, 0.0, 0.5, 0.0, 9.001953125, 1.0, 0.5, 0.0]
proteinsDisplay.OpacityTransferFunction.Points = [9.0, 0.0, 0.5, 0.0, 9.001953125, 1.0, 0.5, 0.0]

cHOLDisplay = Show(cHOL, renderView1, 'UnstructuredGridRepresentation')
cHOLDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.831, 0.627, 0.09],
    ColorArrayName=['FIELD', ''],
    DiffuseColor=[0.831, 0.627, 0.09],
    MapScalars=0,
    Opacity=0.85,
    GaussianRadius=2.0,
)
cHOLDisplay.ScaleTransferFunction.Points = [2.0, 0.0, 0.5, 0.0, 2.00048828125, 1.0, 0.5, 0.0]
cHOLDisplay.OpacityTransferFunction.Points = [2.0, 0.0, 0.5, 0.0, 2.00048828125, 1.0, 0.5, 0.0]

tails_TubeDisplay = Show(tails_Tube, renderView1, 'GeometryRepresentation')

electron_densityLUT = GetColorTransferFunction('electron_density')
electron_densityLUT.Set(
    RGBPoints=[
        0.25, 0.0, 0.2, 1.0,
        0.3987999975681304, 0.5, 0.7, 1.0,
        0.46257142509732924, 1.0, 0.9, 0.0,
        0.49799999594688416, 1.0, 0.0, 0.0,
    ],
    ColorSpace='RGB',
    ScalarRangeInitialized=1.0,
)

tails_TubeDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    SelectNormalArray='TubeNormals',
)
tails_TubeDisplay.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
tails_TubeDisplay.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

head_outDisplay = Show(head_out, renderView1, 'UnstructuredGridRepresentation')
head_outDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.0, 0.6666666865348816, 0.49803921580314636],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.0, 0.6666666865348816, 0.49803921580314636],
    GaussianRadius=3.8,
)
head_outDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
head_outDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

head_inDisplay = Show(head_in, renderView1, 'UnstructuredGridRepresentation')
head_inDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[1.0, 0.6666666865348816, 0.49803921580314636],
    ColorArrayName=[None, ''],
    DiffuseColor=[1.0, 0.6666666865348816, 0.49803921580314636],
    GaussianRadius=3.8,
)
head_inDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
head_inDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

raft_inDisplay = Show(raft_in, renderView1, 'UnstructuredGridRepresentation')
raft_inDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.7803921699523926, 0.5176470875740051, 0.38823530077934265],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.7803921699523926, 0.5176470875740051, 0.38823530077934265],
    GaussianRadius=3.8,
)
raft_inDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
raft_inDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

glyc_outDisplay = Show(glyc_out, renderView1, 'UnstructuredGridRepresentation')
glyc_outDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.9372549057006836, 0.6235294342041016, 0.46666666865348816],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.9372549057006836, 0.6235294342041016, 0.46666666865348816],
    GaussianRadius=2.5,
)
glyc_outDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
glyc_outDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

raft_outDisplay = Show(raft_out, renderView1, 'UnstructuredGridRepresentation')
raft_outDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.0, 0.4588235318660736, 0.33725491166114807],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.0, 0.4588235318660736, 0.33725491166114807],
    GaussianRadius=3.8,
)
raft_outDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
raft_outDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

glyc_inDisplay = Show(glyc_in, renderView1, 'UnstructuredGridRepresentation')
glyc_inDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.0, 0.7450980544090271, 0.545098066329956],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.0, 0.7450980544090271, 0.545098066329956],
    GaussianRadius=2.5,
)
glyc_inDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
glyc_inDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

pIPsDisplay = Show(pIPs, renderView1, 'UnstructuredGridRepresentation')
pIPsDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.6666666865348816, 0.0, 1.0],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.6666666865348816, 0.0, 1.0],
    GaussianRadius=3.8,
)
pIPsDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
pIPsDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

e_densityDisplay = Show(e_density, renderView1, 'UnstructuredGridRepresentation')
e_densityDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    Opacity=0.35,
    GaussianRadius=2.5,
)
e_densityDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
e_densityDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

electron_densityLUTColorBar = GetScalarBar(electron_densityLUT, renderView1)
electron_densityLUTColorBar.Set(
    Title='electron_density',
    ComponentTitle='',
)
electron_densityLUTColorBar.Visibility = 1

tails_TubeDisplay.SetScalarBarVisibility(renderView1, True)
e_densityDisplay.SetScalarBarVisibility(renderView1, True)

electron_densityPWF = GetOpacityTransferFunction('electron_density')
electron_densityPWF.Set(
    Points=[0.25, 0.0, 0.5, 0.0, 0.49799999594688416, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

timeKeeper1 = GetTimeKeeper()
timeAnimationCue1 = GetTimeTrack()
animationScene1 = GetAnimationScene()
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
)

SetActiveSource(e_density)

# RenderAllViews()
# Interact()
# SaveScreenshot("path/to/screenshot.png")
