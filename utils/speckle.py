# ============================================================
# OCTA Speckle Variance Angiogram Pipeline
#
# Pipeline:
#   1. Load 400 sequential OCT B-scans
#   2. Compute speckle variance (= OCTA signal)
#      High variance between frames = moving RBCs = vessels
#      Low variance = static tissue = background
#   3. 3D volume rendering in Slicer (interactive)
#   4. 2D en face projection (max intensity along depth)
#   5. CLAHE enhancement for visualisation
#
# Physical basis:
#   SNR improves as sqrt(NR) where NR = number of repeat scans
#   400 sequential B-scans at different Y positions give us
#   a full 3D volume — speckle variance across neighbouring
#   B-scans extracts the flow signal at each voxel
#
# Usage: paste into 3D Slicer Python Console
# ============================================================

slicer.util.pip_install('opencv-python')
import os, numpy as np, cv2
from PIL import Image
import slicer, vtk

# ── Config ────────────────────────────────────────────────────
bscan_dir  = r"C:\\Miscellaneous\\Ongoing\\URA_GRAD_JOB\\photomedicine-labs\data\\andrei\Bscans"
output_dir = r"C:\\Miscellaneous\\Ongoing\\URA_GRAD_JOB\\photomedicine-labs\data\\andrei"

# ── 1. Load B-scans ───────────────────────────────────────────
files = sorted([f for f in os.listdir(bscan_dir) if f.endswith('.png')])
print(f"Found {len(files)} B-scans")

stack = np.array([
    np.array(Image.open(os.path.join(bscan_dir, f)).convert('L'))
    for f in files
], dtype=np.float32)
print(f"Volume shape: {stack.shape}")
# Shape: (400 B-scans, 2048 depth pixels, 400 A-lines)

# ── 2. Compute speckle variance (OCTA signal) ─────────────────
# 5-frame window: variance across 5 consecutive B-scans per voxel
# Wider window = better SNR but assumes slower motion between frames
print("Computing speckle variance (5-frame window)...")
sv = np.zeros_like(stack)
for i in range(2, len(stack)-2):
    sv[i] = np.var(stack[i-2:i+3], axis=0)

sv_norm = ((sv - sv.min()) / (sv.max() - sv.min()) * 255).astype(np.uint8)

# ── 3. Push 3D volume into Slicer ────────────────────────────
imageData = vtk.vtkImageData()
imageData.SetDimensions(sv_norm.shape[2], sv_norm.shape[1], sv_norm.shape[0])
imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
arr = vtk.util.numpy_support.numpy_to_vtk(
    sv_norm.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
)
imageData.GetPointData().SetScalars(arr)

volumeNode = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLScalarVolumeNode", "OCTA_SpeckleVariance"
)
volumeNode.SetAndObserveImageData(imageData)
volumeNode.CreateDefaultDisplayNodes()
slicer.util.setSliceViewerLayers(background=volumeNode)

# Volume rendering
vrLogic  = slicer.modules.volumerendering.logic()
dispNode = vrLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
dispNode.SetVisibility(True)
volProp  = dispNode.GetVolumePropertyNode().GetVolumeProperty()

# Opacity transfer function — suppress noise, show vessels
opacityFunc = volProp.GetScalarOpacity()
opacityFunc.RemoveAllPoints()
opacityFunc.AddPoint(0,   0.0)
opacityFunc.AddPoint(80,  0.0)   # noise floor cutoff
opacityFunc.AddPoint(110, 0.05)
opacityFunc.AddPoint(180, 0.5)
opacityFunc.AddPoint(255, 0.95)

# Colour: black → dark red → orange → white (angiogram look)
colorFunc = volProp.GetRGBTransferFunction()
colorFunc.RemoveAllPoints()
colorFunc.AddRGBPoint(0,   0.0, 0.0, 0.0)
colorFunc.AddRGBPoint(30,  0.6, 0.1, 0.0)
colorFunc.AddRGBPoint(100, 1.0, 0.5, 0.0)
colorFunc.AddRGBPoint(255, 1.0, 1.0, 0.8)

slicer.util.resetThreeDViews()
print("3D volume loaded — rotate the 3D view to explore")

# ── 4. En face projection (2D angiogram) ─────────────────────
# Max intensity projection collapses depth axis
# Brightest vessel signal at each XY position = en face OCTA map
mip = np.max(sv, axis=1)  # (400, 400) — top-down view
mip = ((mip - mip.min()) / (mip.max() - mip.min()) * 255).astype(np.uint8)

enface_path = os.path.join(output_dir, 'OCTA_enface.png')
Image.fromarray(mip).save(enface_path)
print(f"Saved en face projection: {enface_path}")

# ── 5. CLAHE enhancement ──────────────────────────────────────
# Contrast Limited Adaptive Histogram Equalization
# Enhances local vessel contrast without amplifying noise globally
# clipLimit=2.0, tileGridSize=(8,8) = standard for OCTA imaging
clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(mip)

enhanced_path = os.path.join(output_dir, 'OCTA_enface_enhanced.png')
Image.fromarray(enhanced).save(enhanced_path)
print(f"Saved enhanced en face: {enhanced_path}")

print("\n=== DONE ===")
print("Files saved:")
print(f"  {enface_path}")
print(f"  {enhanced_path}")
print("  - OCTA_enface.png         = raw speckle variance projection")
print("  - OCTA_enface_enhanced.png = CLAHE enhanced (vessels more visible)")
print("  - 3D view in Slicer       = interactive volume rendering")
print("  - Horizontal stripe       = bulk motion artifact (Liao et al. target)")