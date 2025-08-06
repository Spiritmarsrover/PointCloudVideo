
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class CameraDepth : UdonSharpBehaviour
{
    public Camera renderCam; // Assign this in the Inspector

    //public RenderTexture colorRT;
    //public RenderTexture depthRT;
    //public Material targetMat; // Material using the split shader
    //public int width=1280;
    //public int height=720;
    private void Update()
    {
        //int width = Screen.width;
        //int height = Screen.height;

        // Create the color RenderTexture (no depth here)
        //colorRT = //new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
        //colorRT.Create();

        // Create the depth RenderTexture, explicitly readable by shaders
        //depthRT = colorRT//new RenderTexture(width, height, 24, RenderTextureFormat.Depth);
        //depthRT.Create();

        // Assign manually using SetTargetBuffers
        //renderCam.targetTexture = null; // reset just in case
        //renderCam.SetTargetBuffers(colorRT.colorBuffer, depthRT.depthBuffer);


        // Assign to material
        //targetMat.SetTexture("_TopTex", colorRT);          // Top: color
        //targetMat.SetTexture("_BottomDepthTex", depthRT);  // Bottom: depth

        renderCam.depthTextureMode = DepthTextureMode.Depth;
        //renderCam.tex = DepthTextureMode.Depth;

    }

}
