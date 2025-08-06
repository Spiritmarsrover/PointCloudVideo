Shader "Custom/PointCloudV2"
{
    Properties
    {
        [Header(Basic Settings)]
        _MainTex ("Texture (Color top half, Depth bottom half)", 2D) = "white" {}
        _PointSize ("Point Size", Range(0.001, 0.1)) = 0.01
        _PointDensity ("Point Density", Range(1, 10)) = 1
        [Toggle] _Debug ("Debug View", Float) = 0
          [Space(10)]
        [Header(Depth Settings)]
        _DepthScale ("Depth Scale", Range(0.1, 10.0)) = 1.0
        _MinDepth ("Minimum Depth", Range(0.0, 0.1)) = 0.001
        [ToggleUI] _DepthIsSRGB ("Compensate for sRGB Texture", Float) = 1
        _DepthMinimum ("Depth Minimum", Range(-10.0, 1.0)) = 0.0
        _DepthMaximum ("Depth Maximum", Range(0.0, 1.0)) = 1.0
        
        [Space(10)]
        [Header(Camera Settings)]
        _FxRatio ("Focal Length X Ratio", Range(0.1, 2.0)) = 0.7
        _FyRatio ("Focal Length Y Ratio", Range(0.1, 2.0)) = 0.7
        
        [Space(10)]
        [Header(Data Decoding)]
        [ToggleUI] _TileEncodingGamma ("Apply Gamma for Encoding", Float) = 0
        _EnabledSlot ("Enabled Slot", Int) = 0
        
        [Space(10)]
        [Header(Position Control)]
        [ToggleUI] _UseDecodedPosition ("Use Decoded Position", Float) = 0
        _XPosSlot ("X Position Slot", Int) = 1
        _YPosSlot ("Y Position Slot", Int) = 2
        _ZPosSlot ("Z Position Slot", Int) = 3
        _PosScaleSlot ("Position Scale Slot", Int) = 4
        _PositionConstant ("Position Constant (m)", Float) = 10.0
        
        [Space(10)]
        [Header(Rotation Control)]
        [ToggleUI] _UseDecodedRotation ("Use Decoded Rotation", Float) = 0
        _QuatXSlot ("Quaternion X Slot", Int) = 5
        _QuatYSlot ("Quaternion Y Slot", Int) = 6
        _QuatZSlot ("Quaternion Z Slot", Int) = 7
        _QuatWSlot ("Quaternion W Slot", Int) = 8
        
        [Space(10)]
        [Header(Focal Length Control)]
        [ToggleUI] _UseDecodedFocal ("Use Decoded Focal Lengths", Float) = 0
        _FxSlot ("Focal Length X Slot", Int) = 11
        _FySlot ("Focal Length Y Slot", Int) = 12
        
        [Space(10)]
        [Header(Depth Range Control)]
        [ToggleUI] _UseDecodedDepthRange ("Use Decoded Depth Range", Float) = 0
        _DepthMinSlot ("Depth Min Slot", Int) = 9
        _DepthMaxSlot ("Depth Max Slot", Int) = 10
        _DepthConstant ("Depth Constant (m)", Float) = 10.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 100
        ZWrite On
        ZTest LEqual
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            #pragma target 4.0
            
            #include "UnityCG.cginc"
            #include "Codec.hlsl"
            #include "VideoLayout.hlsl"
            
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2g
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 worldPos : TEXCOORD1;
            };
            
            struct g2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 col : COLOR;
            };

            Texture2D_half _MainTex;
            SamplerState _MainTex_sampler;
            float4 _MainTex_ST;
            float4 _MainTex_TexelSize;
            float _PointSize;
            float _DepthScale;
            float _MinDepth;
            int _PointDensity;
            float _FxRatio;
            float _FyRatio;
            float _Debug;
            float _DepthIsSRGB;
            float _TileEncodingGamma;
            float _DepthMinimum;
            float _DepthMaximum;
            
            // Data slot indices
            int _EnabledSlot;
            int _XPosSlot;
            int _YPosSlot;
            int _ZPosSlot;
            int _PosScaleSlot;
            int _QuatXSlot;
            int _QuatYSlot;
            int _QuatZSlot;
            int _QuatWSlot;
            int _FxSlot;
            int _FySlot;
            int _DepthMinSlot;
            int _DepthMaxSlot;
            
            // Toggles and constants
            float _DepthConstant;
            float _PositionConstant;
            float _UseDecodedFocal;
            float _UseDecodedDepthRange;
            float _UseDecodedPosition;
            float _UseDecodedRotation;
            
            // Convert linear value back to gamma (to counteract Unity's automatic gamma-to-linear conversion)
            float LinearToGammaDepth(float x) {
                return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0/2.4) - 0.055;
            }
            
            // Helper functions to get color and depth UVs from the stacked texture
            float2 GetColorUV(float2 uv) {
                // Map to top half (y: 0.5-1.0)
                return float2(uv.x, uv.y * 0.5 + 0.5);
            }
            
            float2 GetDepthUV(float2 uv) {
                // Map to bottom half (y: 0.0-0.5)
                return float2(uv.x, uv.y * 0.5);
            }
            
            // Helper function to sample and decode a value from the specified slot
            float DecodePositionFromSlot(uint slotIdx)
            {
                // Get the tile rectangle for this slot
                float4 rect = GetTileRect(slotIdx);
                
                // Sample the tile colors
                ColorTile c;
                SampleTile(c, _MainTex, rect * _MainTex_ST.xyxy + _MainTex_ST.zwzw, _TileEncodingGamma > 0.5);
                
                // Decode the value (-1 to 1 range)
                return DecodeVideoSnorm(c);
            }
            
            // Helper function to rotate a vector by a quaternion
            float3 RotateVectorByQuaternion(float3 v, float4 q)
            {
                // q = (x, y, z, w)
                float3 qv = q.xyz;
                float qs = q.w;
                
                // Formula: v' = v + 2 * cross(qv, cross(qv, v) + qs * v)
                return v + 2.0 * cross(qv, cross(qv, v) + qs * v);
            }
            
            v2g vert (appdata v)
            {
                v2g o;
                o.vertex = v.vertex;
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            
            [maxvertexcount(100)] // Limited by DX11 restrictions
            void geom(triangle v2g input[3], inout TriangleStream<g2f> triStream)
            {
                // Check if encoding is enabled by reading slot 0
                float encodingEnabled = DecodePositionFromSlot(_EnabledSlot);
                bool useEncoding = (encodingEnabled > 0.5); // Consider enabled if > 0.5
                
                // Initialize quaternion as identity
                float4 rotationQ = float4(0, 0, 0, 1); // Identity quaternion
                
                // Initialize focal length ratios with shader properties
                float fxRatio = _FxRatio;
                float fyRatio = _FyRatio;
                  // Initialize depth range values
                float depthMin = _DepthMinimum;
                float depthMax = _DepthMaximum;
                
                // Initialize position values
                float3 posOffset = float3(0, 0, 0);
                float posScale = 1.0;
                
                if (useEncoding) {
                    // Decode position if enabled
                    if (_UseDecodedPosition > 0.5) {
                        // Decode position from slots 1-3
                        float xPos = DecodePositionFromSlot(_XPosSlot);
                        float yPos = DecodePositionFromSlot(_YPosSlot);
                        float zPos = DecodePositionFromSlot(_ZPosSlot);
                        
                        // Decode position scale from slot 4
                        posScale = max(0.001, DecodePositionFromSlot(_PosScaleSlot));
                        
                        // Calculate position offset
                        posOffset = float3(xPos, yPos, zPos) * posScale * _PositionConstant;
                    }
                    
                    // Decode quaternion from slots
                    if (_UseDecodedRotation > 0.5) {
                        float qx = DecodePositionFromSlot(_QuatXSlot);
                        float qy = DecodePositionFromSlot(_QuatYSlot);
                        float qz = DecodePositionFromSlot(_QuatZSlot);
                        float qw = DecodePositionFromSlot(_QuatWSlot);
                        
                        // Normalize the quaternion
                        float qLen = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
                        if (qLen > 0.0001) {
                            rotationQ = float4(qx/qLen, qy/qLen, qz/qLen, qw/qLen);
                        }
                    }
                    
                    // Decode focal length ratios if enabled
                    if (_UseDecodedFocal > 0.5) {
                        // Decode values are in 0-1 range
                        float decodedFx = DecodePositionFromSlot(_FxSlot);
                        float decodedFy = DecodePositionFromSlot(_FySlot);
                        
                        // Apply the decoded values if they're valid (positive)
                        fxRatio = (decodedFx > 0) ? decodedFx : fxRatio;
                        fyRatio = (decodedFy > 0) ? decodedFy : fyRatio;
                    }
                    
                    // Decode depth range if enabled
                    if (_UseDecodedDepthRange > 0.5) {
                        depthMin = clamp(DecodePositionFromSlot(_DepthMinSlot), 0.0, 1.0);
                        depthMax = clamp(DecodePositionFromSlot(_DepthMaxSlot), 0.0, 1.0);
                        
                        // Ensure depth range is sensible (min < max)
                        if (depthMin > depthMax) {
                            float temp = depthMin;
                            depthMin = depthMax;
                            depthMax = temp;
                        }
                    }
                }
                
                // Debug mode just passes through the original triangle
                if (_Debug > 0.5) {
                    g2f o;
                    [unroll]
                    for (int i = 0; i < 3; i++) {
                        o.pos = UnityObjectToClipPos(input[i].vertex);
                        o.uv = input[i].uv;
                        // Sample from color section (top half)
                        o.col = _MainTex.SampleLevel(LinearClamp, GetColorUV(input[i].uv), 0);
                        triStream.Append(o);
                    }
                    triStream.RestartStrip();
                    return;
                }
                
                // Calculate actual focal lengths based on image dimensions
                float fx = fxRatio * _MainTex_TexelSize.z; // fx = ratio * width
                float fy = fyRatio * _MainTex_TexelSize.w; // fy = ratio * height
                
                // Calculate principal point (assumed to be at image center)
                float cx = _MainTex_TexelSize.z * 0.5;
                float cy = _MainTex_TexelSize.w * 0.5;
                
                // For point cloud, sample across the triangle
                float step = 1.0 / _PointDensity;
                
                // Interpolate across the triangle to create points
                for (float u = 0; u <= 1.0; u += step) {
                    for (float v = 0; v <= 1.0 - u; v += step) {
                        float w = 1.0 - u - v;
                        
                        // Barycentric interpolation of vertex positions and UVs
                        float4 vertPos = input[0].vertex * u + input[1].vertex * v + input[2].vertex * w;
                        float2 uv = input[0].uv * u + input[1].uv * v + input[2].uv * w;
                        
                        // Skip if UV is outside valid range
                        if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
                            continue;
                        
                        // Sample depth from bottom half of texture
                        float depthSample = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv), 0).r;
                        
                        // If the texture is sRGB, convert sampled value back to gamma to compensate
                        // for Unity's automatic gamma-to-linear conversion
                        #if !defined(UNITY_COLORSPACE_GAMMA)
                        if (_DepthIsSRGB > 0.5) {
                            // When in linear rendering mode and sRGB texture is enabled, 
                            // Unity automatically converts texture samples to linear space.
                            // We need to convert back to get the original depth value.
                            depthSample = LinearToGammaDepth(depthSample);
                        }
                        #endif
                          // Calculate depth based on mode
                        float depth;
                        if (_UseDecodedDepthRange > 0.5 && useEncoding) {
                            // Apply depth range formula:
                            // (depth value * (depth max - depth min) + depth min) * depth constant
                            depth = (depthSample * (depthMax - depthMin) + depthMin) * _DepthConstant;
                        } else {
                            // Use manual depth range settings
                            depth = (depthSample * (depthMax - depthMin) + depthMin) * _DepthConstant;
                        }
                        
                        // Skip points with insufficient depth
                        if (depth < _MinDepth)
                            continue;
                        
                        // Sample color from top half of texture
                        float4 color = _MainTex.SampleLevel(LinearClamp, GetColorUV(uv), 0);
                        
                        // Skip fully transparent pixels
                        if (color.a < 0.01)
                            continue;
                        
                        // Convert UV to pixel coordinates
                        float2 pixelCoord = float2(
                            uv.x * _MainTex_TexelSize.z,
                            uv.y * _MainTex_TexelSize.w
                        );
                        
                        // Apply perspective projection formula to calculate 3D position
                        // X = (x_pixel - cx) * Z / fx
                        // Y = (y_pixel - cy) * Z / fy
                        // Z = depth
                        float3 position = float3(
                            (pixelCoord.x - cx) * depth / fx,
                            (pixelCoord.y - cy) * depth / fy,
                            depth
                        );
                        
                        // Apply quaternion rotation if enabled
                        if (useEncoding && _UseDecodedRotation > 0.5) {
                            position = RotateVectorByQuaternion(position, rotationQ);
                        }
                        
                        // Apply position offset if enabled
                        if (_UseDecodedPosition > 0.5 && useEncoding) {
                            position += posOffset;
                        }
                        
                        // Transform to world space
                        float3 worldPos = mul(unity_ObjectToWorld, float4(position, 1.0)).xyz;
                        
                        // Calculate distance to camera for point size scaling
                        float3 viewDir = worldPos - _WorldSpaceCameraPos;
                        float distToCam = length(viewDir);
                        float pointScale = _PointSize * distToCam * 0.1; // Scale based on distance
                        
                        // Create a billboard facing the camera
                        float3 up = float3(0, 1, 0);
                        float3 right = normalize(cross(normalize(viewDir), up));
                        up = normalize(cross(right, normalize(viewDir)));
                        
                        // Create billboard vertices in world space
                        g2f vert;
                        vert.col = color;
                        vert.uv = uv;
                        
                        // Bottom left
                        float3 vertWorldPos = worldPos - right * pointScale - up * pointScale;
                        vert.pos = mul(UNITY_MATRIX_VP, float4(vertWorldPos, 1.0));
                        triStream.Append(vert);
                        
                        // Bottom right
                        vertWorldPos = worldPos + right * pointScale - up * pointScale;
                        vert.pos = mul(UNITY_MATRIX_VP, float4(vertWorldPos, 1.0));
                        triStream.Append(vert);
                        
                        // Top left
                        vertWorldPos = worldPos - right * pointScale + up * pointScale;
                        vert.pos = mul(UNITY_MATRIX_VP, float4(vertWorldPos, 1.0));
                        triStream.Append(vert);
                        
                        // Top right
                        vertWorldPos = worldPos + right * pointScale + up * pointScale;
                        vert.pos = mul(UNITY_MATRIX_VP, float4(vertWorldPos, 1.0));
                        triStream.Append(vert);
                        
                        triStream.RestartStrip();
                    }
                }
            }
            
            fixed4 frag (g2f i) : SV_Target
            {
                return i.col;
            }
            ENDCG
        }
    }
    FallBack "Diffuse" // Fallback for when geometry shaders aren't supported
}