import os as _os
_vtp_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    'bicapa_sim0005.vtp'
)

# state file generated using paraview version 6.1.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 1

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views
# ----------------------------------------------------------------

# Layout #1 — vista lateral superior (thumbnail)
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1561, 540]

# Layout #2 — vista principal perspectiva con densidad electronica
renderView2 = CreateView('RenderView')
renderView2.Set(
    ViewSize=[1561, 786],
    CenterOfRotation=[250.4515562057495, 250.52066135406494, 0.0],
    CameraPosition=[235.1928196653456, -741.1319940481455, 132.93890274839072],
    CameraFocalPoint=[257.1151358839198, 683.5811880132964, -58.055197980958475],
    CameraViewUp=[0.051680895982342476, 0.1319105292728045, 0.9899134796826594],
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup layouts
# ----------------------------------------------------------------

layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView2)
layout1.SetSize(1561, 786)

layout1_1 = CreateLayout(name='Layout #1')
layout1_1.AssignView(0, renderView1)
layout1_1.SetSize(1561, 540)

SetActiveView(renderView2)

# ----------------------------------------------------------------
# selection sources
# ----------------------------------------------------------------

selection_sources23004 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.23004', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)&(leaflet == 0)',
    Assembly='',
    Selectors=['/'])

selection_sources26087 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.26087', groupname='selection_sources', ElementType='Point Data',
    QueryString='(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_sources23049 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.23049', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)&(leaflet == 1)',
    Assembly='',
    Selectors=['/'])

selection_sources25799 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.25799', groupname='selection_sources', ElementType='Point Data',
    QueryString='(pip_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter23037 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.23037', groupname='selection_sources', Input=selection_sources23004,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources13619 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.13619', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter13652 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.13652', groupname='selection_sources', Input=selection_sources13619,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter26120 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.26120', groupname='selection_sources', Input=selection_sources26087,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources13079 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.13079', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)',
    Assembly='',
    Selectors=['/'])

selection_filter13124 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.13124', groupname='selection_sources', Input=selection_sources13079,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter25832 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.25832', groupname='selection_sources', Input=selection_sources25799,
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

selection_sources13906 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.13906', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 4)',
    Assembly='',
    Selectors=['/'])

selection_filter13939 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.13939', groupname='selection_sources', Input=selection_sources13906,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter23082 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.23082', groupname='selection_sources', Input=selection_sources23049,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24680 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24680', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 1)&(leaflet == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24713 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24713', groupname='selection_sources', Input=selection_sources24680,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24479 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24479', groupname='selection_sources', ElementType='Point Data',
    QueryString='(in_raft == 1)&(leaflet == 1)&(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24512 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24512', groupname='selection_sources', Input=selection_sources24479,
    Expression='s0',
    SelectionNames=['s0'])

# ----------------------------------------------------------------
# selections
# ----------------------------------------------------------------

selectionSource0 = CreateSelection(proxyname='SelectionQuerySource', registrationname='SelectionSource0', groupname='selections', ElementType='Point Data',
    QueryString='(region == 4)',
    Assembly='',
    Selectors=['/'])

appendSelections = CreateSelection(proxyname='AppendSelections', registrationname='AppendSelections', groupname='selections', Input=selectionSource0,
    Expression='s0',
    SelectionNames=['s0'])

# ----------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------

general = XMLPolyDataReader(registrationName='General', FileName=[_vtp_path])
general.Set(
    PointArrayStatus=['region', 'is_head', 'is_glycerol', 'is_tail', 'is_protein', 'order_param', 'in_raft', 'is_pip', 'pip_head', 'electron_density', 'lipid_id', 'leaflet', 'bead_type', 'seg_idx', 'n_doublebonds', 'phase', 'chain_length'],
    TimeArray='None',
)

tails_Source = XMLPolyDataReader(registrationName='Tails_Source', FileName=[_vtp_path])
tails_Source.Set(
    PointArrayStatus=['region', 'in_raft', 'electron_density', 'phase'],
    TimeArray='None',
)

head_out = ExtractSelection(registrationName='Head_out', Input=general,
    Selection=selection_filter23037)

tails = Threshold(registrationName='Tails', Input=tails_Source)
tails.Set(
    Scalars=['POINTS', 'region'],
    LowerThreshold=2.0,
    UpperThreshold=3.0,
)

headEDensity = ExtractSelection(registrationName='Head E.Density', Input=general,
    Selection=selection_filter13124)

tails_Poly = ExtractSurface(registrationName='Tails_Poly', Input=tails)

tails_Tube = Tube(registrationName='Tails_Tube', Input=tails_Poly)
tails_Tube.Set(
    Scalars=['POINTS', ''],
    Vectors=['POINTS', ''],
    NumberofSides=12,
    Radius=1.2,
)

glyc_in = ExtractSelection(registrationName='Glyc_in', Input=general,
    Selection=selection_filter24624)

raft_out = ExtractSelection(registrationName='Raft_out', Input=general,
    Selection=selection_filter24557)

cholEDensity = ExtractSelection(registrationName='Chol E.Density', Input=general,
    Selection=selection_filter13939)

cHOL = Threshold(registrationName='CHOL', Input=general)
cHOL.Set(
    Scalars=['POINTS', 'region'],
    LowerThreshold=4.0,
    UpperThreshold=4.0,
)

proteins = Threshold(registrationName='Proteins', Input=general)
proteins.Set(
    Scalars=['POINTS', 'is_protein'],
    LowerThreshold=1.0,
    UpperThreshold=1.0,
)

head_in = ExtractSelection(registrationName='Head_in', Input=general,
    Selection=selection_filter23082)

pIPs = ExtractSelection(registrationName='PIPs', Input=head_in,
    Selection=selection_filter25832)

glycEDensity = ExtractSelection(registrationName='Glyc E.Density', Input=general,
    Selection=selection_filter13652)

glyc_out = ExtractSelection(registrationName='Glyc_out', Input=general,
    Selection=selection_filter24713)

raft_in = ExtractSelection(registrationName='Raft_in', Input=general,
    Selection=selection_filter24512)

appendSelections.SetSelectionId(general.GetGlobalID())
appendSelections.SetSelectionPort(0)

# ----------------------------------------------------------------
# visualization in renderView2
# ----------------------------------------------------------------

proteinsDisplay = Show(proteins, renderView2, 'UnstructuredGridRepresentation')
proteinsDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.615686297416687, 0.615686297416687, 0.615686297416687],
    ColorArrayName=['FIELD', ''],
    DiffuseColor=[0.615686297416687, 0.615686297416687, 0.615686297416687],
    MapScalars=0,
    Opacity=0.6,
    GaussianRadius=5.0,
)
proteinsDisplay.ScaleTransferFunction.Points = [9.0, 0.0, 0.5, 0.0, 9.001953125, 1.0, 0.5, 0.0]
proteinsDisplay.OpacityTransferFunction.Points = [9.0, 0.0, 0.5, 0.0, 9.001953125, 1.0, 0.5, 0.0]

tails_TubeDisplay = Show(tails_Tube, renderView2, 'GeometryRepresentation')

electron_densityLUT = GetColorTransferFunction('electron_density')
electron_densityLUT.Set(
    RGBPoints=GenerateRGBPoints(
        preset_name='Blue - Green - Orange',
        range_min=0.25,
        range_max=0.49799999594688416,
    ),
    ColorSpace='RGB',
    NanColor=[0.25, 0.0, 0.0],
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

headEDensityDisplay = Show(headEDensity, renderView2, 'UnstructuredGridRepresentation')
headEDensityDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    GaussianRadius=3.8,
)
headEDensityDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
headEDensityDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

glycEDensityDisplay = Show(glycEDensity, renderView2, 'UnstructuredGridRepresentation')
glycEDensityDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    GaussianRadius=3.0,
)
glycEDensityDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
glycEDensityDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

cholEDensityDisplay = Show(cholEDensity, renderView2, 'UnstructuredGridRepresentation')
cholEDensityDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    GaussianRadius=2.4,
)
cholEDensityDisplay.ScaleTransferFunction.Points = [2.0, 0.0, 0.5, 0.0, 2.00048828125, 1.0, 0.5, 0.0]
cholEDensityDisplay.OpacityTransferFunction.Points = [2.0, 0.0, 0.5, 0.0, 2.00048828125, 1.0, 0.5, 0.0]

electron_densityLUTColorBar = GetScalarBar(electron_densityLUT, renderView2)
electron_densityLUTColorBar.Set(
    Title='electron_density',
    ComponentTitle='',
)
electron_densityLUTColorBar.Visibility = 1

tails_TubeDisplay.SetScalarBarVisibility(renderView2, True)
headEDensityDisplay.SetScalarBarVisibility(renderView2, True)
glycEDensityDisplay.SetScalarBarVisibility(renderView2, True)
cholEDensityDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# opacity transfer function
# ----------------------------------------------------------------

electron_densityPWF = GetOpacityTransferFunction('electron_density')
electron_densityPWF.Set(
    Points=[0.25, 0.6517857313156128, 0.5, 0.0, 0.3783155381679535, 0.424107164144516, 0.5, 0.0, 0.49799999594688416, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# animation scene
# ----------------------------------------------------------------

timeAnimationCue1 = GetTimeTrack()
timeKeeper1 = GetTimeKeeper()
animationScene1 = GetAnimationScene()
animationScene1.Set(
    ViewModules=renderView2,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
)

SetActiveSource(raft_out)

# RenderAllViews()
# Interact()
# SaveScreenshot("path/to/screenshot.png")
# SaveScreenshot("path/to/screenshot.png", GetLayout())
