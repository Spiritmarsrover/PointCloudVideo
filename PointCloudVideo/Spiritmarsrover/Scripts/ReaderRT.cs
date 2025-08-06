
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class ReaderRT : UdonSharpBehaviour
{
    public Texture2D outputTexture;

    private RenderTexture inputTexture;
    private Color[] colors;


    //public Texture2D dataTexture;
    //public Animator animator;
    public int layer;
   // public bool applyScale = true;
   // public bool applyVisibility = true;
    // public bool onlyHips = false;

    //private Transform root;
   // private Transform[] bones;
   // private Renderer[] renderers;


    void Start()
    {
        
    }
    
    void OnEnable()
    {
        inputTexture = GetComponent<Camera>().targetTexture;
        colors = new Color[inputTexture.width * inputTexture.height];
    }

    void OnPostRender()
    {
        VRC.SDK3.Rendering.VRCAsyncGPUReadback.Request(inputTexture, 0, TextureFormat.RGBAFloat, (VRC.Udon.Common.Interfaces.IUdonEventReceiver)(Component)this);
    }

    public void OnAsyncGpuReadbackComplete(VRC.SDK3.Rendering.VRCAsyncGPUReadbackRequest request)
    {
        request.TryGetData(colors);
        outputTexture.SetPixels(colors);
    }
    private void Update()
    {
        var offsetY = layer * 4;
        var c1 = (Vector3)(Vector4)outputTexture.GetPixel(0, offsetY + 1);
        var c3 = (Vector3)(Vector4)outputTexture.GetPixel(0, offsetY + 3);
        var rescale = c1.magnitude;
        var valid = !float.IsNaN(c3.magnitude) && rescale > 0f;
        //gameObject.transform.position = c3;
        Debug.Log("PosOut: " + c3);
    }

}
