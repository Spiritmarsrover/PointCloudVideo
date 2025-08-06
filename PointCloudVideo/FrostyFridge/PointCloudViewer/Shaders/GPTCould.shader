Shader "Custom/PointCloud_UnityCamera"
{
	Properties
	{
		[Header(Basic Settings)]
		_MainTex("Texture (Color top half, Depth bottom half)", 2D) = "white" {}
		_PointSize("Point Size", Range(0.0001, 0.5)) = 0.01
		_PointDensity("Point Density", Range(1, 10)) = 1
		[Toggle] _Debug("Debug View", Float) = 0
		  [Space(10)]
		[Header(Depth Settings)]
		_DepthScale("Depth Scale", Range(0.1, 10.0)) = 1.0
		_MinDepth("Minimum Depth", Range(0.0, 0.1)) = 0.001
		[ToggleUI] _DepthIsSRGB("Compensate for sRGB Texture", Float) = 1
		_DepthMinimum("Depth Minimum", Range(-10.0, 1.0)) = 0.0
		_DepthMaximum("Depth Maximum", Range(0.0, 1.0)) = 1.0

		[Space(10)]
		[Header(Camera Settings)]
		_FxRatio("Focal Length X Ratio", Range(0.1, 2.0)) = 0.7
		_FyRatio("Focal Length Y Ratio", Range(0.1, 2.0)) = 0.7
		[Space(5)]
		_FOV("Camera FOV", Range(0.0, 170.0)) = 60.0
		_Aspect("Camera Aspect Ratio", Range(0.0, 3.0)) = 1.7777777
		//_Far("Camera Far Plane", Range(0.0, 10000.0)) = 1000.0
		//_Near("Camera Near Plane", Range(0.0, 10000.0)) = 0.3
		_DepthGamma("Depth Gamma", Range(0.0, 20)) = 1
		_Keyr("KeyR", Range(0.0, 1)) = 1
		_Keyg("KeyG", Range(0.0, 1)) = 1
		_Keyb("KeyB", Range(0.0, 1)) = 1

		_KernelSize("_KernelSize", Range(0.0, 0.003)) = 0.003
		_Threashold("_Threashold", Range(0.0, .25)) = .25
		[ToggleUI] _BackgroundProjectionRemoval("Remove backgroundProjection", Float) = 1

		//[ToggleUI]_Ortho("Is Ortho?", Float) = 0
		_OrthoSize("Ortho Size", Range(0.0, 10)) = 1

		[Space(10)]
		[Header(Data Decoding)]
		[ToggleUI] _TileEncodingGamma("Apply Gamma for Encoding", Float) = 0
		_EnabledSlot("Enabled Slot", Int) = 0

		[Space(10)]
		[Header(Position Control)]
		[ToggleUI] _UseDecodedPosition("Use Decoded Position", Float) = 0
		_XPosSlot("X Position Slot", Int) = 1
		_YPosSlot("Y Position Slot", Int) = 2
		_ZPosSlot("Z Position Slot", Int) = 3
		_PosScaleSlot("Position Scale Slot", Int) = 4
		_PositionConstant("Position Constant (m)", Float) = 10.0

		[Space(10)]
		[Header(Rotation Control)]
		[ToggleUI] _UseDecodedRotation("Use Decoded Rotation", Float) = 0
		_QuatXSlot("Quaternion X Slot", Int) = 5
		_QuatYSlot("Quaternion Y Slot", Int) = 6
		_QuatZSlot("Quaternion Z Slot", Int) = 7
		_QuatWSlot("Quaternion W Slot", Int) = 8

		[Space(10)]
		[Header(Focal Length Control)]
		[ToggleUI] _UseDecodedFocal("Use Decoded Focal Lengths", Float) = 0
		_FxSlot("Focal Length X Slot", Int) = 11
		_FySlot("Focal Length Y Slot", Int) = 12

		[Space(10)]
		[Header(Depth Range Control)]
		[ToggleUI] _UseDecodedDepthRange("Use Decoded Depth Range", Float) = 0

		_NEARhiSlot("NEARhiSlot", Int) = 9
		_NEARloSlot("NEARloSlot", Int) = 10

		_FARhiSlot("FARhiSlot", Int) = 10
		_FARloSlot("FARloSlot", Int) = 10
		
		//_DepthConstant("Depth Constant (m)", Float) = 10.0
	}
		SubShader
		{
			Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
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
				float _FOV;
				float _Aspect;
				float _Far;
				float _Near;
				float _Debug;
				float _Ortho;
				float _OrthoSize;
				float _DepthGamma;
				float _Keyr;
				float _Keyg;
				float _Keyb;
				float _KernelSize;
				float _Threashold;
				float _BackgroundProjectionRemoval;

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

				float FOVSIZE;//used for both so we can save a slot

				int _NEARhiSlot;
				int _NEARloSlot;

				int _FARhiSlot;
				int _FARloSlot;

				// Convert linear value back to gamma (to counteract Unity's automatic gamma-to-linear conversion)
				float LinearToGammaDepth(float x) {
					return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
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

				v2g vert(appdata v)
				{
					v2g o;
					o.vertex = v.vertex;
					o.worldPos = mul(unity_ObjectToWorld, v.vertex);
					o.uv = TRANSFORM_TEX(v.uv, _MainTex);
					return o;
				}
				float LinearEyeDepth(float z, float near, float far)
				{
					return near / (1.0 - z * (1.0 - near / far));
				}
				float LinearEyeDepth2(float z, float near, float far) {
					return (2.0 * near * far) / (far + near - z * (far - near));
				}
				float LinearEyeDepthOrtho(float rawDepth, float near, float far)
				{
					return lerp(near, far, rawDepth);
				}
				// ---- Morton Encode Helper ----
				int InterleaveBits(int v)
				{
					v &= 0xF; // 4-bit
					v = (v | (v << 8)) & 0x0F00F;
					v = (v | (v << 4)) & 0xC30C3;
					v = (v | (v << 2)) & 0x49249;
					return v;
				}
				//
				int Morton3D_Encode(int x, int y, int z)
				{
					return InterleaveBits(x) | (InterleaveBits(y) << 1) | (InterleaveBits(z) << 2);
				}
				float3 RGBtoYUV(float3 rgb)
				{
					float3 yuv;
					yuv.x = dot(rgb, float3(0.2126, 0.7152, 0.0722));               // Y
					yuv.y = dot(rgb, float3(-0.1146, -0.3854, 0.5)) + 0.5;           // U
					yuv.z = dot(rgb, float3(0.5, -0.4542, -0.0458)) + 0.5;           // V
					return saturate(yuv);
				}

				float3 RGBtoYUV_BT709(float3 rgb) {
					float Y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
					float U = -0.1146 * rgb.r - 0.3854 * rgb.g + 0.5000 * rgb.b + 0.5;
					float V = 0.5000 * rgb.r - 0.4542 * rgb.g - 0.0458 * rgb.b + 0.5;
					return float3(Y, U, V);
				}
				float DecodeDepthFromRGB(float3 rgb, int yBits, int uvBits)
				{
					float3 yuv = RGBtoYUV_BT709(rgb);

					int yPart = (int)(saturate(yuv.x) * ((1 << yBits) - 1) + 0.5);
					int uPart = (int)(saturate(yuv.y) * ((1 << (uvBits / 2)) - 1) + 0.5);
					int vPart = (int)(saturate(yuv.z) * ((1 << (uvBits / 2)) - 1) + 0.5);

					int uvPart = (uPart << (uvBits / 2)) | vPart;
					int intDepth = (yPart << uvBits) | uvPart;

					int totalBits = yBits + uvBits;
					float depth = intDepth / float((1 << totalBits) - 1);
					return depth;
				}

				float DecodeDepthFromRGB_YUVOrder(float3 rgb)
				{
					// Convert from RGB back to YUV
					float3 yuv = RGBtoYUV_BT709(rgb);

					// Get 8 bits from each channel
					int high = (int)(saturate(yuv.x) * 255.0 + 0.5);
					int mid = (int)(saturate(yuv.y) * 255.0 + 0.5);
					int low = (int)(saturate(yuv.z) * 255.0 + 0.5);

					// Reconstruct 24-bit integer
					int dInt = (high << 16) | (mid << 8) | low;

					// Normalize to 0–1
					return (float)dInt / 16777215.0;
				}
				float DecodeDepth16Bit(float3 rgb)
				{
					uint r = (uint)(rgb.g * 15.0 + 0.5);   // 4 bits
					uint g = (uint)(rgb.r * 255.0 + 0.5);  // 8 bits
					uint b = (uint)(rgb.b * 15.0 + 0.5);   // 4 bits

					uint value = (g << 8) | (r << 4) | b;  // reconstruct 16-bit depth
					return value / 65535.0;
				}
				//Jet like colormap
				float ColorMapDecode(float3 colord) {
					float depth = 1.0;
					float small = 0.01;
					float big = 0.9890;

					if (colord.r > small&&colord.g > small&&colord.b > small&&colord.r < big&&colord.g < big&&colord.b < big)//didn't fall on the colormap
					{
						if (colord.r > colord.g&&colord.r > colord.b)
						{
							colord.r = 1.0;
							if (colord.g > colord.b)
							{
								colord.g = 1.0;
							}
							else
							{
								colord.b = 1.0;
							}
						}
						if (colord.g > colord.r&&colord.g > colord.b)
						{
							colord.g = 1.0;
							if (colord.r > colord.b)
							{
								colord.r = 1.0;
							}
							else
							{
								colord.b = 1.0;
							}
						}
						if (colord.b > colord.r&&colord.b > colord.g)
						{
							colord.b = 1.0;
							if (colord.r > colord.g)
							{
								colord.r = 1.0;
							}
							else
							{
								colord.g = 1.0;
							}
						}
					}

					if (colord.r < 1.0 && colord.g < small && colord.b < small)
					{
						// Segment 1: Red from 0 to 1
						depth = colord.r * (1.0 / 7.0) + (0.0 / 7.0);
					}
					else if (colord.r > big && colord.g < 1.0 && colord.b < small)
					{
						// Segment 2: Green from 0 to 1
						depth = colord.g * (1.0 / 7.0) + (1.0 / 7.0);
					}
					else if (colord.r < 1.0 && colord.g > big && colord.b < small)
					{
						// Segment 3: Red decreases from 1 to 0
						depth = (1.0 - colord.r) * (1.0 / 7.0) + (2.0 / 7.0);
					}
					else if (colord.r < small && colord.g > big && colord.b < 1.0)
					{
						// Segment 4: Blue increases from 0 to 1
						depth = colord.b * (1.0 / 7.0) + (3.0 / 7.0);
					}
					else if (colord.r < 1.0 && colord.g > big && colord.b > big)
					{
						// Segment 5: Red increases from 0 to 1
						depth = colord.r * (1.0 / 7.0) + (4.0 / 7.0);
					}
					else if (colord.r > big && colord.g < 1.0 && colord.b > big)
					{
						// Segment 6: Green decreases from 1 to 0
						depth = (1.0 - colord.g) * (1.0 / 7.0) + (5.0 / 7.0);
					}
					else if (colord.r < 1.0 && colord.g < small && colord.b > big)
					{
						// Segment 7: Red decreases from 1 to 0
						depth = (1.0 - colord.r) * (1.0 / 7.0) + (6.0 / 7.0);
					}
					return depth;
				}

				// Morton encoding (3D to 1D Z-order curve)
				uint MortonEncode3D(uint x, uint y, uint z) {
					uint answer = 0;
					for (uint i = 0; i < 8; ++i) {
						answer |= ((x >> i) & 1) << (3 * i);
						answer |= ((y >> i) & 1) << (3 * i + 1);
						answer |= ((z >> i) & 1) << (3 * i + 2);
					}
					return answer;
				}

				// Morton decoding (1D to 3D Z-order curve)
				void MortonDecode3D(uint code, out uint x, out uint y, out uint z) {
					x = y = z = 0;
					for (uint i = 0; i < 8; ++i) {
						x |= ((code >> (3 * i)) & 1) << i;
						y |= ((code >> (3 * i + 1)) & 1) << i;
						z |= ((code >> (3 * i + 2)) & 1) << i;
					}
				}
				// -------- CONFIGURABLE PARAM --------
				#define BUCKET_DIVISOR 12
				#define MAX_MORTON_CODE (BUCKET_DIVISOR * BUCKET_DIVISOR * BUCKET_DIVISOR)
				#define BUCKET_SIZE (256 / BUCKET_DIVISOR)
				// -------- DECODER --------
				float DecodeRGBToDepth(float3 rgb) {
					uint x = min((uint)(rgb.r * 255.0 / BUCKET_SIZE), BUCKET_DIVISOR - 1);
					uint y = min((uint)(rgb.g * 255.0 / BUCKET_SIZE), BUCKET_DIVISOR - 1);
					uint z = min((uint)(rgb.b * 255.0 / BUCKET_SIZE), BUCKET_DIVISOR - 1);

					uint depthIndex = MortonEncode3D(x, y, z);
					float depth = (float)depthIndex / (float)(MAX_MORTON_CODE - 1);
					return depth;
				}
				float DecodeRGBToDepth12bit(float3 rgb) {
					// Convert from [0,1] to 0–15 integers
					int r = (int)(rgb.r * 15.0 + 0.5);
					int g = (int)(rgb.g * 15.0 + 0.5);
					int b = (int)(rgb.b * 15.0 + 0.5);

					// Reconstruct the 12-bit depth index
					int d = (g << 8) | (r << 4) | b;

					// Convert back to normalized 0–1 depth
					return d / 4095.0;
				}
				// Decode the RGB back into a 0–1 depth
				float DecodeDepthFromRGB12M(float3 rgb)
				{
					// 1) Reconstruct the 4bit indices from each channel
					uint x = (uint)floor(saturate(rgb.r) * 15.0 + 0.5);
					uint y = (uint)floor(saturate(rgb.g) * 15.0 + 0.5);
					uint z = (uint)floor(saturate(rgb.b) * 15.0 + 0.5);

					// 2) Interleave their bits back into one 12bit Morton index
					uint idx = 0;
					for (uint i = 0; i < 4; ++i)
					{
						idx |= ((x >> i) & 1u) << (3 * i);
						idx |= ((y >> i) & 1u) << (3 * i + 1);
						idx |= ((z >> i) & 1u) << (3 * i + 2);
					}

					// 3) Convert back to [0..1]
					return idx / 4095.0;
				}

				float DecodeRGBToDepth_12bit(float3 rgb)
				{
					// undo limitedrange mapping
					float inv255 = 1.0 / 255.0;
					float low = 16.0 * inv255;
					float high = 235.0 * inv255;
					float rn = saturate((rgb.r - low) / (high - low));
					float gn = saturate((rgb.g - low) / (high - low));

					uint r = (uint)floor(rn * 255.0 + 0.5) >> 4;  // back to 4bit LSB
					uint g = (uint)floor(gn * 255.0 + 0.5);       // 8bit MSB

					uint depthInt = (g << 4) | r;
					return float(depthInt) / 4095.0;
				}
				// Convert a hue [0,1] + saturation+value into RGB
				float3 HSVtoRGB(float h, float s, float v)
				{
					// h wraps [0,1)
					float3 K = float3(1.0, 2.0 / 3.0, 1.0 / 3.0);
					float3 p = abs(frac(h + K) * 6.0 - 3.0);
					float3 rgb = saturate(p - 1.0);
					return v * lerp(1.0, rgb, s);
				}

				// Convert an RGB color back to HSV; returns (h,s,v)
				float3 RGBtoHSV(float3 c)
				{
					float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
					float4 p = lerp(
						float4(c.bg, K.wz),
						float4(c.gb, K.xy),
						step(c.b, c.g)
					);
					float4 q = lerp(
						float4(p.xyw, c.r),
						float4(c.r, p.yzx),
						step(p.x, c.r)
					);
					float d = q.x - min(q.w, q.y);
					float e = 1e-10;
					float h = abs(q.z + (q.w - q.y) / (6.0 * d + e));
					return float3(h, d / (q.x + e), q.x);
				}
				// Decode RGB colormap  depth  [0,1]
				float DecodeColorToDepth(float3 rgb)
				{
					// invert gamma from Rec.709 “limited” (approx sRGB)
					// linearize for more accurate hue extraction
					rgb = pow(rgb, 1.0);
					float3 hsv = RGBtoHSV(rgb);
					// Hue was in [0,0.8], so divide back
					return saturate(hsv.x / 0.8);
				}


				// Quantization settings
				#define DEPTH_BINS 4096        // 12-bit depth
				#define INV_DEPTH_BINS (1.0 / 4096.0)

				// Approximate Oklab to Linear sRGB conversion
				float3 OklabToLinear(float3 lab) {
					float l = lab.x, a = lab.y, b = lab.z;

					float l_ = pow(l + 0.3963377774 * a + 0.2158037573 * b, 3.0);
					float m_ = pow(l - 0.1055613458 * a - 0.0638541728 * b, 3.0);
					float s_ = pow(l - 0.0894841775 * a - 1.2914855480 * b, 3.0);

					float3 rgb = float3(
						+4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
						-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
						-0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
						);

					return rgb;
				}

				// sRGB encode
				float3 LinearToSRGB(float3 rgb) {
					rgb = saturate(rgb);
					return pow(rgb, 1.0 / 2.2); // approximate gamma correction
				}

				float3 SRGBToLinear(float3 srgb) {
					return pow(saturate(srgb), 2.2);
				}

				float3 EncodeDepthToRGB_Oklab(float depth) {
					// Quantize
					int idx = (int)(depth * (DEPTH_BINS - 1));

					// Map index to a perceptual color ramp in Oklab space
					float t = idx * INV_DEPTH_BINS; // normalized 0 to 1

					// Spiral-like ramp in Oklab (perceptual spiral)
					float L = 0.6 + 0.3 * sin(t * 6.283);      // Keep in middle luminance range
					float A = 0.15 * cos(t * 6.283 * 3.0);     // A/B oscillations
					float B = 0.15 * sin(t * 6.283 * 3.0);

					float3 lab = float3(L, A, B);

					// Convert to sRGB
					float3 linearRGB = OklabToLinear(lab);
					return LinearToSRGB(linearRGB);
				}
				// Precomputed Oklab color ramp on CPU side would be ideal. But for now, brute-force decode://No chat gpt you cant brute for this 
				//float DecodeDepthFromRGB_Oklab(float3 srgb) {
				//	float3 linearRGB = SRGBToLinear(srgb);

				//	// Convert linear RGB to Oklab (approximate reverse)
				//	// We’ll brute-force match against known index set (can also store a LUT if needed)

				//	float bestDiff = 1e8;
				//	int bestIndex = 0;

				//	for (int i = 0; i < DEPTH_BINS; ++i) {
				//		float t = i * INV_DEPTH_BINS;

				//		float L = 0.6 + 0.3 * sin(t * 6.283);
				//		float A = 0.15 * cos(t * 6.283 * 3.0);
				//		float B = 0.15 * sin(t * 6.283 * 3.0);
				//		float3 lab = float3(L, A, B);

				//		float3 rgbGuess = LinearToSRGB(OklabToLinear(lab));
				//		float diff = distance(rgbGuess, srgb); // Euclidean distance

				//		if (diff < bestDiff) {
				//			bestDiff = diff;
				//			bestIndex = i;
				//		}
				//	}

				//	return bestIndex * INV_DEPTH_BINS;
				//}
				float DecodeLoopHSV(float3 c)
				{
					// tiny epsilon for comparisons
					const float e = 1e-5;
					float v = 0;

					if (abs(c.r - 1) < e && abs(c.b) < e)
						// segment 0: r=1, b=0, g=t
						v = c.g / 6.0;
					else if (abs(c.g - 1) < e && abs(c.b) < e)
						// segment 1: g=1, b=0, r=1–t
						v = (1 - c.r) / 6.0 + 1.0 / 6.0;
					else if (abs(c.r) < e && abs(c.g - 1) < e)
						// segment 2: r=0, g=1, b=t
						v = c.b / 6.0 + 2.0 / 6.0;
					else if (abs(c.r) < e && abs(c.b - 1) < e)
						// segment 3: r=0, b=1, g=1–t
						v = (1 - c.g) / 6.0 + 3.0 / 6.0;
					else if (abs(c.g) < e && abs(c.b - 1) < e)
						// segment 4: g=0, b=1, r=t
						v = c.r / 6.0 + 4.0 / 6.0;
					else if (abs(c.r - 1) < e && abs(c.g) < e)
						// segment 5: r=1, g=0, b=1–t
						v = (1 - c.b) / 6.0 + 5.0 / 6.0;
					else
						// Not exactly on the loop – clamp to [0,1]
						v = 0;

					return saturate(v);
				}

				float DecodeLoopHSV2(float3 c)
				{
					float e = 0.1; // Expanded tolerance radius
					float v = -1.0;
					//float small = 0.0;

					if (c.r > 0.9 && c.b < 0.1 && c.g <0.9)			//(1,0,0) -> (1,1,0)
					{
						v = c.g / (0.9*6.0);
					}
					else if (c.g > 0.9 && c.b < 0.1 && c.r > 0.1)	//(1,1,0) -> (0,1,0)
					{
						v = (1.0 - c.r) / (0.9 * 6.0) + (1.0 / 6.0);
					}
					else if (c.r < 0.1 && c.g > 0.9 && c.b < 0.9)	//(0,1,0) -> (0,1,1)
					{
						v = (c.b) / (0.9 * 6.0) + (2.0 / 6.0);
					}
					else if (c.r < 0.1 && c.g > 0.1 && c.b > 0.9)	//(0,1,1) -> (0,0,1)
					{
						v = (1.0 - c.g) / (0.9 * 6.0) + (3.0 / 6.0);
					}
					else if (c.r < 0.9 && c.g < 0.1 && c.b > 0.9)	//(0,0,1) -> (1,0,1)
					{
						v = ( c.r) / (0.9 * 6.0) + (4.0 / 6.0);
					}
					else if (c.r > 0.9 && c.g < 0.1 && c.b > 0.1)	//(1,0,1) -> (1,0,0)
					{
						v = (1.0 - c.b) / (0.9 * 6.0) + (5.0 / 6.0);
					}
					/*else if (c.g > 0.9 && c.b < 0.1 && c.r < 1.0)
					{
						v = (1 - c.r) / (0.9 * 6.0) + 1.0 / 6.0;
					}*/
					//if (abs(c.r - 1.0) < e && abs(c.b) < e)
					//	v = (c.g) / (0.9 * 6.0);
					//else if (abs(c.g - 1.0) < e && abs(c.b) < e)
					//	v = (1 - c.r) / (0.9 * 6.0) + 1.0 / 6.0;
					//else if (abs(c.r) < e && abs(c.g - 1.0) < e)
					//	v = (c.b) / (0.9 * 6.0) + 2.0 / 6.0;
					//else if (abs(c.r) < e && abs(c.b - 1.0) < e)
					//	v = (1 - c.g) / (0.9 * 6.0) + 3.0 / 6.0;
					//else if (abs(c.g) < e && abs(c.b - 1.0) < e)
					//	v = (c.r) / (0.9 * 6.0) + 4.0 / 6.0;
					//else if (abs(c.r - 1.0) < e && abs(c.g) < e)
					//	v = (1 - c.b) / (0.9 * 6.0) + 5.0 / 6.0;
					//else
					//	v = -1.0; // Outside all ranges

					return saturate(v);
				}

				float depthDecode(float3 colord) {
					////NAIVE ENCODE
					/*float value;
					value = (colord.r * 0.33333) + (colord.g * 0.33333) + (colord.b * 0.33333);
					return 1.0-value;*/

					//Morton
					//// Convert RGB from 0–1 to 0–255, then divide by 16 to get 0–15 voxel index
					//int x = int(floor(colord.r * 255.0 + 0.5)) / 16;
					//int y = int(floor(colord.g * 255.0 + 0.5)) / 16;
					//int z = int(floor(colord.b * 255.0 + 0.5)) / 16;

					//int morton = Morton3D_Encode(x, y, z);
					//return (morton + 0.5) / 4096.0;


					//Jet or jet like encode for some more depth data.
					//float3 color = float3(0.0, 0.0, 0.0);

					//if (depth >= 0.0 && depth <= (1.0 / 7.0))
					//{
					//	color.r = (depth - (0.0 / 7.0)) / (1.0 / 7.0);
					//	color.g = 0.0;
					//	color.b = 0.0;
					//}

					//if (depth > (1.0 / 7.0) && depth <= (2.0 / 7.0))
					//{
					//	color.r = 1.0;
					//	color.g = (depth - (1.0 / 7.0)) / (1.0 / 7.0);
					//	color.b = 0.0;
					//}

					//if (depth > (2.0 / 7.0) && depth <= (3.0 / 7.0))
					//{
					//	color.r = 1.0 - (depth - (2.0 / 7.0)) / (1.0 / 7.0);
					//	color.g = 1.0;
					//	color.b = 0.0;
					//}

					//if (depth > (3.0 / 7.0) && depth <= (4.0 / 7.0))
					//{
					//	color.r = 0.0;
					//	color.g = 1.0;
					//	color.b = (depth - (3.0 / 7.0)) / (1.0 / 7.0);
					//}

					//if (depth > (4.0 / 7.0) && depth <= (5.0 / 7.0))
					//{
					//	color.r = (depth - (4.0 / 7.0)) / (1.0 / 7.0);
					//	color.g = 1.0;
					//	color.b = 1.0;
					//}

					//if (depth > (5.0 / 7.0) && depth <= (6.0 / 7.0))
					//{
					//	color.r = 1.0;
					//	color.g = 1.0 - (depth - (5.0 / 7.0)) / (1.0 / 7.0);
					//	color.b = 1.0;
					//}

					//if (depth > (6.0 / 7.0) && depth <= (7.0 / 7.0))
					//{
					//	color.r = 1.0 - (depth - (6.0 / 7.0)) / (1.0 / 7.0);
					//	color.g = 0.0;
					//	color.b = 1.0;
					//}

					////return color;


					//float depthout = 0.0;

					//if (colord.r > 0.0 && colord.g == 0.0 && colord.b == 0.0)
					//{
					//	depthout = colord.r*(1.0 / 7.0) + 0.0;
					//}
					//if (colord.r == 1.0 && colord.g > 0.0 && colord.b == 0.0)
					//{
					//	depthout = colord.g*(1.0 / 7.0) + (1.0 / 7.0);
					//	//depthout = 0.05;
					//}

					//24bit encode
					// Input: RGB color in[0, 1]
					/*uint r = (uint)(colord.r * 255.0 + 0.5);
					uint g = (uint)(colord.g * 255.0 + 0.5);
					uint b = (uint)(colord.b * 255.0 + 0.5);

					uint depthInt = (r << 16) | (g << 8) | b;
					float depth = depthInt / 16777215.0;
					return depth;*/

					//Dyanmic 24-n bit depth decode
					//const uint totalBits = 24;
					//uint msbBits = clamp(8, 8, 24);
					//uint lsbBits = totalBits - msbBits;

					////float3 yuv = RGBtoYUV(colord);
					//float3 yuv = RGBtoYUV_BT709(colord);

					//uint msb = (uint)(yuv.x * ((1u << msbBits) - 1) + 0.5);
					//uint lsb = 0;

					//if (lsbBits > 0)
					//{
					//	uint halfBits = lsbBits / 2;

					//	uint lsbHigh = (uint)(((yuv.y - 0.25) / 0.5) * ((1u << halfBits) - 1) + 0.5);
					//	uint lsbLow = (uint)(((yuv.z - 0.25) / 0.5) * ((1u << halfBits) - 1) + 0.5);

					//	lsb = (lsbHigh << halfBits) | lsbLow;
					//}

					//uint depthInt = (msb << lsbBits) | lsb;
					//return (float)depthInt / 16777215.0;

					//return depthout;

					////JET WRaping
					//float depth = 1.0;

					//if (colord.r < 1.0 && colord.g == 0.0 && colord.b == 0.0)
					//{
					//	// Segment 1: Red from 0 to 1
					//	depth = colord.r * (1.0 / 7.0) + (0.0 / 7.0);
					//}
					//else if (colord.r == 1.0 && colord.g < 1.0 && colord.b == 0.0)
					//{
					//	// Segment 2: Green from 0 to 1
					//	depth = colord.g * (1.0 / 7.0) + (1.0 / 7.0);
					//}
					//else if (colord.r < 1.0 && colord.g == 1.0 && colord.b == 0.0)
					//{
					//	// Segment 3: Red decreases from 1 to 0
					//	depth = (1.0 - colord.r) * (1.0 / 7.0) + (2.0 / 7.0);
					//}
					//else if (colord.r == 0.0 && colord.g == 1.0 && colord.b < 1.0)
					//{
					//	// Segment 4: Blue increases from 0 to 1
					//	depth = colord.b * (1.0 / 7.0) + (3.0 / 7.0);
					//}
					//else if (colord.r < 1.0 && colord.g == 1.0 && colord.b == 1.0)
					//{
					//	// Segment 5: Red increases from 0 to 1
					//	depth = colord.r * (1.0 / 7.0) + (4.0 / 7.0);
					//}
					//else if (colord.r == 1.0 && colord.g < 1.0 && colord.b == 1.0)
					//{
					//	// Segment 6: Green decreases from 1 to 0
					//	depth = (1.0 - colord.g) * (1.0 / 7.0) + (5.0 / 7.0);
					//}
					//else if (colord.r < 1.0 && colord.g == 0.0 && colord.b == 1.0)
					//{
					//	// Segment 7: Red decreases from 1 to 0
					//	depth = (1.0 - colord.r) * (1.0 / 7.0) + (6.0 / 7.0);
					//}
					//return depth;
					//return DecodeDepthFromRGB(colord,8,8);
					//return DecodeDepthFromRGB_YUVOrder(colord);
					//return DecodeDepth16Bit(colord);
					//return 1.0-JetDecode(colord);
					//return 1.0- DecodeRGBToDepth(colord);
					//return 1.0- DecodeRGBToDepth12bit(colord);
					//return 1.0- DecodeDepthFromRGB_Oklab(colord);//dont do this
					//return 1.0- ColorMapDecode(colord);
					//return 1.0- DecodeLoopHSV(colord);
					return 1.0- DecodeLoopHSV2(colord);
					//return 1.0-DecodeDepth_FromRGBBucket(colord);
					//return colord.r + colord.g / 255.0 + colord.b / (255.0 * 255.0);
					//return colord.r + colord.g / 255.0 + colord.b / (255.0 * 255.0);
				}

				[maxvertexcount(100)] // Limited by DX11 restrictions
				void geom(triangle v2g input[3], inout TriangleStream<g2f> triStream)
				{
					// Check if encoding is enabled by reading slot 0
					//float encodingEnabled = DecodePositionFromSlot(_EnabledSlot);//why do this???
					float encodingEnabled = 1.0;
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
						  //slots 0 and 3 define x high and low floats
						  float highx = DecodePositionFromSlot(_XPosSlot);
						  float lowx = DecodePositionFromSlot(_XPosSlot+3);
						  //slots 1 and 4 define y high and low floats
						  float highy = DecodePositionFromSlot(_YPosSlot);
						  float lowy = DecodePositionFromSlot(_YPosSlot+3);
						  //slots 2 and 5 define z high and low floats
						  float highz = DecodePositionFromSlot(_ZPosSlot);
						  float lowz = DecodePositionFromSlot(_ZPosSlot+3);
						  //get the actual floats from the encode (+-1430 or 730?)
						  float xPos = DecodeVideoFloat(highx, lowx);
						  float yPos = DecodeVideoFloat(highy, lowy);
						  float zPos = DecodeVideoFloat(highz, lowz);
						  // Decode position from slots 1-3
						  /*float xPos = DecodePositionFromSlot(_XPosSlot);
						  float yPos = DecodePositionFromSlot(_YPosSlot);
						  float zPos = DecodePositionFromSlot(_ZPosSlot);*/

						  // Decode position scale from slot 4
						  //posScale = max(0.001, DecodePositionFromSlot(_PosScaleSlot));

						  // Calculate position offset
						  //posOffset = float3(xPos, yPos, zPos) * posScale * _PositionConstant;

						  //posOffset = float3(xPos, yPos, zPos);// *posScale * _PositionConstant;//no need for these i think
						  posOffset = float3(xPos, yPos, zPos)*2.0;// *posScale * _PositionConstant;//no need for these i think
					  }

					  // Decode quaternion from slots
					  if (_UseDecodedRotation > 0.5) {
						  float qx = DecodePositionFromSlot(_QuatXSlot);
						  float qy = DecodePositionFromSlot(_QuatYSlot);
						  float qz = DecodePositionFromSlot(_QuatZSlot);
						  float qw = DecodePositionFromSlot(_QuatWSlot);

						  // Normalize the quaternion
						  float qLen = sqrt(qx*qx + qy * qy + qz * qz + qw * qw);
						  if (qLen > 0.0001) {
							  rotationQ = float4(qx / qLen, qy / qLen, qz / qLen, qw / qLen);
						  }
					  }

					  // Decode focal length ratios if enabled
					  if (_UseDecodedFocal > 0.5) {
						  //// Decode values are in 0-1 range
						  //float decodedFx = DecodePositionFromSlot(_FxSlot);
						  //float decodedFy = DecodePositionFromSlot(_FySlot);

						  //// Apply the decoded values if they're valid (positive)
						  //fxRatio = (decodedFx > 0) ? decodedFx : fxRatio;
						  //fyRatio = (decodedFy > 0) ? decodedFy : fyRatio;
						  float decodedFOVSIZEhi = DecodePositionFromSlot(_FxSlot);
						  
						  float decodedFOVSIZElo = DecodePositionFromSlot(_FySlot);
						  FOVSIZE = DecodeVideoFloat(decodedFOVSIZEhi, decodedFOVSIZElo);
						  _Ortho = DecodePositionFromSlot(16);
					  }

					  // Decode depth range if enabled
					  if (_UseDecodedDepthRange > 0.5) {
						  //depthMin = clamp(DecodePositionFromSlot(_DepthMinSlot), 0.0, 1.0);
						  //depthMax = clamp(DecodePositionFromSlot(_DepthMaxSlot), 0.0, 1.0);

						  //// Ensure depth range is sensible (min < max)
						  //if (depthMin > depthMax) {
							 // float temp = depthMin;
							 // depthMin = depthMax;
							 // depthMax = temp;
						  //}
						  float decodedNearhi = DecodePositionFromSlot(_NEARhiSlot);
						  float decodedNearlo = DecodePositionFromSlot(_NEARloSlot);
						  
						  float decodedFarhi = DecodePositionFromSlot(_FARhiSlot);
						  float decodedFarlo = DecodePositionFromSlot(_FARloSlot);

						  _Near = DecodeVideoFloat(decodedNearhi, decodedNearlo);
						  _Far = DecodeVideoFloat(decodedFarhi, decodedFarlo);

						  //_Far = 2.0;

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
				  //float fovRadians = radians(_FOV);
				  float fovRadians = radians(FOVSIZE);
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
						//float depth;
						//if (_UseDecodedDepthRange > 0.5 && useEncoding) {
						//	// Apply depth range formula:
						//	// (depth value * (depth max - depth min) + depth min) * depth constant
						//	depth = (depthSample * (depthMax - depthMin) + depthMin) * _DepthConstant;
						//}
						//else
						//{
						//	// Use manual depth range settings
						//	depth = (depthSample * (depthMax - depthMin) + depthMin) * _DepthConstant;
						//}

						//// Skip points with insufficient depth
						//if (depth < _MinDepth)
						//	continue;
						

						/*float near = _Near;
						float far = _Far;*/
						float near = _Near;
						float far = _Far;

						//Used for the UNITY CAMERA
						//float depthCOPE = LinearEyeDepth(_MainTex.SampleLevel(LinearClamp, GetDepthUV(uv), 0).r,near,far);//LinearEyeDepth
						/*float depthCOPE = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv), 0).r;
						float depthG = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv), 0).g;
						float depthB = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv), 0).b;

						float depthCOPE = depthR *.3;*/
						float3 colord = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv), 0).rgb;
						float depthCOPE = depthDecode(colord);

						//if (colord.r >= colord.g && colord.r >= colord.b) {
						//	// R was active (depth in 0.0–0.33333)
						//	depthCOPE = colord.r * 0.33333;
						//}
						//else if (colord.g >= colord.b) {
						//	// G was active (depth in 0.33333–0.66666)
						//	depthCOPE = colord.g * 0.33333 + 0.33333;
						//}
						//else {
						//	// B was active (depth in 0.66666–1.0)
						//	depthCOPE = colord.b * 0.33334 + 0.66666;
						//}



						//float depthCOPE = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv).r;//since its b/w anyways
						float kernelsize = _KernelSize;

						/*float leftDepth = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(-kernelsize*1.7777777, 0)), 0).r;
						float rightDepth = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(kernelsize*1.7777777, 0)), 0).r;
						float upDepth = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(0, kernelsize)), 0).r;
						float downDepth = _MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(0, -kernelsize)), 0).r;*/
						
						float leftDepth = depthDecode(_MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(-kernelsize*1.7777777, 0)), 0).rgb);
						float rightDepth = depthDecode(_MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(kernelsize*1.7777777, 0)), 0).rgb);
						float upDepth = depthDecode(_MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(0, kernelsize)), 0).rgb);
						float downDepth = depthDecode(_MainTex.SampleLevel(LinearClamp, GetDepthUV(uv + float2(0, -kernelsize)), 0).rgb);
						float depthgamma = _DepthGamma;
						//depthCOPE = pow(depthCOPE, 1.0 / depthgamma);//i dont htink we need gamma anymore



						float maxDiff = max(
							abs(depthCOPE - leftDepth),
							max(
								abs(depthCOPE - rightDepth),
								max(abs(depthCOPE - upDepth), abs(depthCOPE - downDepth))
							)
						);
						if (maxDiff > _Threashold)
						{
							continue;
						}
						//cutout the far plane so that it doens't project the background on to the cloud
						if (_BackgroundProjectionRemoval>.5) 
						{
							if (depthCOPE > 0.9999)
							{
								continue;
							}
							if (depthCOPE < 0.001)
							{
								continue;
							}

						}
						
						// Sample color from top half of texture
						float4 color = _MainTex.SampleLevel(LinearClamp, GetColorUV(uv), 0);
						
						if (color.r > _Keyr && color.b > _Keyb && color.g<_Keyg)//chroma key like fucntion to snip out artifacts
						{
							continue;
						}

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

						//FOCAL LENGTH DEPTH
						/*float3 position = float3(
							(pixelCoord.x - cx) * depth / fx,
							(pixelCoord.y - cy) * depth / fy,
							depth
						);*/
						float aspct = _Aspect;
						fy = 0.5*(720.0) / tan(fovRadians*0.5);
						//fx = fy * (1280.0 / 720.0);
						fx = fy * aspct;

						//FOV DEPTH
						/*float3 position = float3(
							(pixelCoord.x - cx) * depthCOPE / fx,
							(pixelCoord.y - cy) * depthCOPE / fy,
							depthCOPE
						);*/

						/*float2 ndc = uv * 2.0 - 1.0;
						float yScale = tan(fovRadians * 0.5);
						float xScale = yScale * aspct;

						float3 viewDir1 = float3(ndc.x * xScale, ndc.y * yScale, 1.0);*/
						float3 position;// = viewDir1 * LinearEyeDepth(depthCOPE, near, far);

						if (!(_Ortho >0.5))
						{
							float2 ndc = uv * 2.0 - 1.0;
							float yScale = tan(fovRadians * 0.5);
							float xScale = yScale * aspct;

							float3 viewDir1 = float3(ndc.x * xScale, ndc.y * yScale, 1.0);
							position = viewDir1 * LinearEyeDepth(depthCOPE, near, far);
						}
						else
						{
							float2 ndc = uv * 2.0 - 1.0;

							//float viewHeight = _OrthoSize * 2.0;
							float viewHeight = FOVSIZE * 2.0;
							float viewWidth = viewHeight * _Aspect;

							float3 viewPos;
							position.x = ndc.x * 0.5 * viewWidth;
							position.y = ndc.y * 0.5 * viewHeight;
							position.z = LinearEyeDepthOrtho(depthCOPE, near, far); // or just depth if it's linear already
						}
						



						//float3 position = viewDir1 * (far-near)*depthCOPE+near;
						// Apply quaternion rotation if enabled
						//if (useEncoding && _UseDecodedRotation > 0.5) 
						if (true) 
						{
							position = RotateVectorByQuaternion(position, rotationQ);
						}

						// Apply position offset if enabled
						//if (_UseDecodedPosition > 0.5 && useEncoding) 
						if (true) 
						{
							position += posOffset;
						}

						// Transform to world space
						float3 worldPos = mul(unity_ObjectToWorld, float4(position, 1.0)).xyz;

						// Calculate distance to camera for point size scaling
						float3 viewDir = worldPos - _WorldSpaceCameraPos;
						float distToCam = length(viewDir);
						//float pointScale = _PointSize * distToCam * 0.1; // Scale based on distance
						//float pointScale = _PointSize / (distToCam);
						//float4 clipPos = UnityObjectToClipPos(position);
						//float pointScale = _PointSize / clipPos.w;
						//float pointScale = _PointSize / max(distToCam, 0.001);
						float pointScale;
						if (_Ortho>0.5)
						{
							pointScale = (FOVSIZE*_PointSize*4) / distToCam * distToCam;
						}
						else
						{
							//pointScale = (radians(FOVSIZE*0.5)*4* _PointSize)  / distToCam * distToCam;
							//pointScale = _PointSize*100 * distToCam * 0.1;
							//float scaleFactor = 1 * distToCam*distToCam * tan(halfFovRad);
							float verticalFOV = FOVSIZE;
							float horizontalFOV = 2.0*atan(tan(radians(FOVSIZE*0.5))*1.7777777);
							float linearDepth = LinearEyeDepth(depthCOPE, _Near, _Far);
							//float adjecendt = ((_Far - _Near)*linearDepth) +_Near;
							float adjecendt = linearDepth;
							float oppisite = (2.0*adjecendt * tan(radians(verticalFOV * 0.5)));//the size of the screen at that depth



							float halfFovRad = radians(FOVSIZE * 0.5);
							pointScale = (3.0*oppisite*_PointSize / distToCam * distToCam);
							//pointScale = (tan(halfFovRad)*_PointSize / distToCam * distToCam);
							
							//pointScale = max(pointScale, 0.01);
						}
						


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

			fixed4 frag(g2f i) : SV_Target
			{
				return i.col;
			}
			ENDCG
		}
		}
			FallBack "Diffuse" // Fallback for when geometry shaders aren't supported
}