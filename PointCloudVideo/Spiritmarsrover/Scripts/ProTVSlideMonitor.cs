
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;
using UnityEngine.UI;

public class ProTVSlideMonitor : UdonSharpBehaviour
{
    public Slider ProTVSlide;
    public GameObject stopbutton;
    public GameObject PointCloud;
    public GameObject DecodingCamera;
    public GameObject ProTVScreen;

    public GameObject recordingCamera;
    void Start()
    {
        PointCloud.transform.position = Vector3.zero;
    }
    private void Update()
    {
        if (ProTVSlide.value < ProTVSlide.maxValue && stopbutton.activeSelf) 
        {
            PointCloud.SetActive(true);
            ProTVScreen.SetActive(true);
            DecodingCamera.SetActive(true);
        }
        else
        {
            PointCloud.SetActive(false);
            ProTVScreen.SetActive(false);
            DecodingCamera.SetActive(false);
        }
        if (recordingCamera.activeSelf)
        {
            DecodingCamera.SetActive(true);
        }
    }

}
