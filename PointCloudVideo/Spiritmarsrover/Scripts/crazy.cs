
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class crazy : UdonSharpBehaviour
{
    //public Transform zero;
    public Material encoded;
    //public float HIGH;
   // public float LOW;
    //int ColorTileRadix = 3;
    //int ColorTileLen = 2;
    //int tilePow = 0;//= (int)Mathf.Pow(ColorTileRadix, ColorTileLen * 3); // 3^6 = 729
                    // public Transform t_test;
   // public Transform t_test;
   // public Vector3 rotYs;
    //public Vector3 rotZs;
    public Camera cam;
    private void Start()
    {
        //tilePow = (int)Mathf.Pow(ColorTileRadix, ColorTileLen * 3); // 3^6 = 729
    }
    //public float DecodeVideoFloat(float hi, float lo)
    //{
    //    float halfRange = (tilePow - 1) / 2f;
    //    Vector2 hilo = new Vector2(hi, lo) * halfRange + new Vector2(halfRange, halfRange);

    //    Vector3 state = Vector3.zero;
    //    GrayEncoderAdd(ref state, hilo.x, tilePow, false);
    //    GrayEncoderAdd(ref state, hilo.y, tilePow, true);

    //    float centeredX = state.x - (tilePow * tilePow - 1) / 2f;
    //    return GrayEncoderSum(new Vector3(centeredX, state.y, state.z)) / halfRange;
    //}

    //void GrayEncoderAdd(ref Vector3 state, float x, int radix, bool cont = true)
    //{
    //    // Reflect value if LSB of state.x is 1
    //    if (((int)state.x & 1) != 0)
    //    {
    //        x = radix - 1 - x;
    //    }

    //    float rounded = Mathf.Round(x);
    //    state.x = state.x * radix + rounded;

    //    if (cont && (rounded == 0f || rounded == (radix - 1)))
    //    {
    //        // Keep state.yz unchanged
    //    }
    //    else
    //    {
    //        float delta = x - rounded;
    //        state.y = delta;
    //        state.z = delta;
    //    }
    //}

    //float GrayEncoderSum(Vector3 state)
    //{
    //    // float2 p = max(0, state.zy * float2(+2, -2));
    //    float px = Mathf.Max(0f, state.z * +2f);
    //    float py = Mathf.Max(0f, state.y * -2f);

    //    float maxXY = Mathf.Max(Mathf.Max(px, py) - px * py, 1e-5f); // Avoid division by zero
    //    float minXY = Mathf.Min(px, py);

    //    float mixPart = minXY / maxXY * (px - py);
    //    float sumPart = (px - py);
    //    return 0.5f * (mixPart + sumPart) + state.x;
    //}


    //// Mimics frac(x) = x - floor(x)
    //static float frac(float x) => x - Mathf.Floor(x);

    //// Decoder step
    //static Vector2 GrayDecoderPop(ref Vector2Int state, uint radix)
    //{
    //    Vector2Int d = new Vector2Int(state.x % (int)radix, state.y % (int)radix);
    //    state.x /= (int)radix;
    //    state.y /= (int)radix;

    //    if ((state.x & 1) != 0) d.x = (int)radix - 1 - d.x;
    //    if ((state.y & 1) != 0) d.y = (int)radix - 1 - d.y;

    //    return new Vector2(d.x, d.y);
    //}

    //// Main encoder
    //public static void EncodeVideoSnorm(float x, out Vector3 leftTile, out Vector3 rightTile, bool hi = false)
    //{
    //    float maxEnc = (730 * 730 - 1) * 0.5f; // = 531441 - 0.5
    //    x = Mathf.Clamp(x * (730 - 1) * 0.5f, -maxEnc, maxEnc);

    //    float fracX = frac(x);
    //    Vector2 wt = new Vector2(1 - fracX, fracX) / (3 - 1);

    //    Vector2Int state = new Vector2Int(
    //        Mathf.FloorToInt(x) + (int)(maxEnc),
    //        Mathf.FloorToInt(x) + (int)(maxEnc) + 1
    //    );

    //    // Encode into 2 Vector3s
    //    Vector3[] tiles = new Vector3[2];

    //    for (int i = 1; i >= 0; i--)
    //    {
    //        Vector2 b = GrayDecoderPop(ref state, 3);
    //        Vector2 r = GrayDecoderPop(ref state, 3);
    //        Vector2 g = GrayDecoderPop(ref state, 3);
    //        tiles[i] = new Vector3(Vector2.Dot(r, wt), Vector2.Dot(g, wt), Vector2.Dot(b, wt));
    //    }

    //    if (hi)
    //    {
    //        for (int i = 1; i >= 0; i--)
    //        {
    //            Vector2 b = GrayDecoderPop(ref state, 3);
    //            Vector2 r = GrayDecoderPop(ref state, 3);
    //            Vector2 g = GrayDecoderPop(ref state, 3);
    //            tiles[i] = new Vector3(Vector2.Dot(r, wt), Vector2.Dot(g, wt), Vector2.Dot(b, wt));
    //        }
    //    }

    //    leftTile = tiles[0];
    //    rightTile = tiles[1];
    //}

    //public Vector3 GetSwingTwistAngles(float[] rot, float eps = 1e-5f)
    //{
    //    // rot is a 9-length float[] representing a 3x3 matrix: c0 (0–2), c1 (3–5), c2 (6–8)
    //    float c0x = rot[0], c0y = rot[1], c0z = rot[2];
    //    float c1y = rot[4], c1z = rot[5];
    //    float c2y = rot[7], c2z = rot[8];

    //    float x = Mathf.Atan2(c1z - c2y, c1y + c2z);

    //    float acosArg = c0x;
    //    float factor;
    //    if (c0x < 1f - eps)
    //    {
    //        float denom = 1f - c0x * c0x;
    //        factor = Mathf.Acos(c0x) / Mathf.Sqrt(denom);
    //    }
    //    else
    //    {
    //        factor = 4f / 3f - c0x / 3f;
    //    }

    //    float y = -c0z * factor;
    //    float z = c0y * factor;

    //    return new Vector3(x, y, z);
    //}
    //Vector3 mul(Vector3 input,Transform t0)
    //{
    //    Vector3 pos = input;
    //    Vector3 c1 = t0.up; // mat0.c1
    //    float dotC1 = Vector3.Dot(c1, c1);

    //    // Transpose of rotation matrix = inverse rotation
    //    pos = Quaternion.Inverse(t0.rotation) * pos;

    //    // Scale
    //    return pos /= dotC1;
    //}
    //Vector3 SwingTwistAngles(Vector3 c0, Vector3 c1, Vector3 c2)
    //{
    //    float swing = Mathf.Atan2(c1.z - c2.y, c1.y + c2.z);

    //    float eps = 1e-5f;
    //    float x = c0.x;
    //    float lenSq = 1 - x * x;

    //    float f;
    //    if (x < 1 - eps && lenSq > eps)
    //    {
    //        f = Mathf.Acos(x) / Mathf.Sqrt(lenSq);
    //    }
    //    else
    //    {
    //        f = 4f / 3f - x / 3f;
    //    }

    //    float swingY = -c0.z * f;
    //    float swingZ = c0.y * f;

    //    return new Vector3(swing, swingY, swingZ);
    //}

    private void Update()
    {
        ////Debug.Log("HIGHLOW: "+DecodeVideoFloat(HIGH, LOW));
        ////Vector3 outputLeft;
        ////Vector3 outputRight;
        ////EncodeVideoSnorm(HIGH, out outputLeft, out outputRight, true);
        ////Debug.Log("Encode: "+ outputLeft+" "+ outputRight);

        ////Vector3 outputLeft1;
        ////Vector3 outputRight1;
        ////EncodeVideoSnorm(LOW, out outputLeft1, out outputRight1, false);
        ////Debug.Log("Encode2: " + outputLeft1 + " " + outputRight1);

        ////Matrix4x4 mat = transform.localToWorldMatrix;
        ////float scale = mat.GetColumn(1).magnitude;
        ////// Get column 1 (c1: local up / Y axis)
        ////Vector3 rotY = mat.GetColumn(1).normalized*Mathf.Min(1f, scale); // or transform.up

        ////// Get column 2 (c2: local forward / Z axis)
        ////Vector3 rotZ = mat.GetColumn(2).normalized * Mathf.Min(1f, 1f/scale); // or transform.forward


        //Transform t1 = transform;
        //Transform t0 = zero; // base pose

        //Matrix4x4 m1 = t1.localToWorldMatrix;
        //Matrix4x4 m0 = t0.localToWorldMatrix;

        //Vector3 pos = m1.GetColumn(3) - m0.GetColumn(3);
        //Vector3 rotY = m1.GetColumn(1);
        //Vector3 rotZ = m1.GetColumn(2);

        //pos = mul(pos,t0);
        //rotY = mul(rotY,t0);
        //rotZ = mul(rotZ,t0);

        //float scale = rotY.magnitude;

        //rotY = rotY.normalized;
        //rotZ = rotZ.normalized;

        ////// Build transpose of mat0 rotation part:
        ////Matrix4x4 m0Rot = Matrix4x4.Rotate(t0.rotation); // rotation only
        ////Matrix4x4 m0RotT = m0Rot.transpose;

        ////float yAxisLenSqr = m0Rot.GetColumn(1).sqrMagnitude; // mat0.c1 dot c1

        //// Transform everything into reference space and normalize by scale
        ////ector3 posInBase = m0RotT.MultiplyVector(pos) / yAxisLenSqr;
        ////Vector3 rotY = m0RotT.MultiplyVector(m1.GetColumn(1)) / yAxisLenSqr;
        ////Vector3 rotZ = m0RotT.MultiplyVector(m1.GetColumn(2)) / yAxisLenSqr;

        //// Store scale (along rotY)
        ////float scale = rotY.magnitude;

        //// Normalize directions
        ////rotY.Normalize();
        ////rotZ.Normalize();
        //Vector3 posInBase = pos;

        //Vector3 rotYScaled = rotY * Mathf.Min(1f, scale);
        //Vector3 rotZScaled = rotZ * Mathf.Min(1f, 1f / scale);
        //// Send to shader (as float4s)
        //encoded.SetVector("_RotY", new Vector3(rotYScaled.x, rotYScaled.y, rotYScaled.z));
        //encoded.SetVector("_RotZ", new Vector3(rotZScaled.x, rotZScaled.y, rotZScaled.z));

        //encoded.SetFloat("_PosX", posInBase.x);
        //encoded.SetFloat("_PosY", posInBase.y);
        //encoded.SetFloat("_PosZ", posInBase.z);

        //rotYs = t_test.localToWorldMatrix.GetColumn(1);
        //rotZs = t_test.localToWorldMatrix.GetColumn(2);
        //Debug.Log("RotYs: " + rotYs);
        //Debug.Log("RotZs: " + rotZs);
        //rotYs = mul(rotYs, t0);
        //rotZs = mul(rotZs, t0);

        //rotYs = rotYs.normalized;
        //rotZs = rotZs.normalized;

        //Vector3 rotXs = Vector3.Cross(rotYs, rotZs);
        //Vector3 angles = SwingTwistAngles(rotXs, rotYs, rotZs);
        //Vector3 data = new Vector3(angles.x / Mathf.PI / 1f, angles.y / Mathf.PI / 1f, angles.z / Mathf.PI / 1f);
        ////encoded.SetVector("_FOVSIZE", data);
        ///


        encoded.SetFloat("_PosX", cam.gameObject.transform.position.x);
        encoded.SetFloat("_PosY", cam.gameObject.transform.position.y);
        encoded.SetFloat("_PosZ", cam.gameObject.transform.position.z);

        encoded.SetFloat("_QuaterionX", cam.gameObject.transform.rotation.x);
        encoded.SetFloat("_QuaterionY", cam.gameObject.transform.rotation.y);
        encoded.SetFloat("_QuaterionZ", cam.gameObject.transform.rotation.z);
        encoded.SetFloat("_QuaterionW", cam.gameObject.transform.rotation.w);

        encoded.SetFloat("_NEAR", cam.nearClipPlane);
        encoded.SetFloat("_FAR", cam.farClipPlane);
        //encoded.SetFloat("_FAR", cam.farClipPlane);

        if (cam.orthographic)
        {
            encoded.SetFloat("_FOVSIZE", cam.orthographicSize);
            encoded.SetFloat("_isOrtho", 1.0f);
        }
        else
        {
            encoded.SetFloat("_FOVSIZE", cam.fieldOfView);
            encoded.SetFloat("_isOrtho", 0.0f);
        }
        

    }
}
