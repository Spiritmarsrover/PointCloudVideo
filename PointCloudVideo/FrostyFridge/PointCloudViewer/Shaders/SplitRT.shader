Shader "Custom/SplitRenderTextureWithDepth"
{
	Properties
	{
		_TopTex("Top Render Texture", 2D) = "white" {}
		_BottomDepthTex("Bottom Depth Texture", 2D) = "black" {}

		_Gamma("Gamma", Range(0.0, 20)) = 2.2
		_DepthBitsInY("Depth Bits", Int)= 8

		//_WidthRatio("Overlay Width", Range(0,1)) = 0.075
		//_Height("Overlay Height", Range(0,1)) = 1.0
		//_GridSize("Grid Size (X=Cols Y=Rows)", Vector) = (6, 45, 0, 0)
		//_VisibleSquares("Number of Active Squares", Int) = 100


		/*_VisibleSlotCount("Visible Slots", Range(0,135)) = 135

		_PosX("X", Float) = 0.0
		_PosY("Y", Float) = 0.0
		_PosZ("Z", Float) = 0.0*/
		

		/*_FOV("Camera FOV", Range(0.0, 170.0)) = 60.0
		_Aspect("Camera Aspect Ratio", Range(0.0, 3.0)) = 1.7777777
		_Far("Camera Far Plane", Range(0.0, 10000.0)) = 1000.0
		_Near("Camera Near Plane", Range(0.0, 10000.0)) = 0.3*/


	}
		SubShader
	{
		Tags { "RenderType" = "Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"
			#include "Codec.hlsl"
			#include "VideoLayout.hlsl"
			
			sampler2D _TopTex;
			sampler2D _BottomDepthTex;
			float _Gamma;

			//float _WidthRatio;
			//float _Height;
			float4 _GridSize;
			//int _VisibleSquares;
			//int _VisibleSlotCount;
			//float _Far;
			//float _Near;\
			
			/*float _PosX;
			float _PosY;
			float _PosZ*/;

			//YUV 24-n bit encoding
			int _DepthBitsInY; // e.g. 16 means 16 bits in Y, 8 in UV


			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				return o;
			}

			//float Linear01Depth(float rawDepth)
			//{
			//	// Optional: assume linear depth already. If reversed-Z or non-linear, change this.
			//	return rawDepth;
			//}
			/*float Linear01Depth(float z, float near, float far)
			{
				return (z * far * near) / ((far - z * (far - near)));
			}*/
			/*float LinearToGammaDepth(float x) {
				return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
			}
			float LinearEyeDepth(float z, float near, float far)
			{
				return near / (1.0 - z * (1.0 - near / far));
			}*/
			float Linear01Depth(float z, float near, float far)
			{
				float linearDepth = near * far / (far - z * (far - near));
				return (linearDepth - near) / (far - near); // Remap to 0–1 linearly
			}
			float LinearEyeDepth(float z, float near, float far)
			{
				return near / (1.0 - z * (1.0 - near / far));
			}
			float LinearToGammaDepth(float x) {
				return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
			}
			//Morthon Encoding testing
			// Interleaves lower 'bits' bits from x, y, z into a Morton code
			//uint Part1By2(uint n) {
			//	n &= 0x000003ff; // only lower 10 bits
			//	n = (n | (n << 16)) & 0x030000FF;
			//	n = (n | (n << 8)) & 0x0300F00F;
			//	n = (n | (n << 4)) & 0x030C30C3;
			//	n = (n | (n << 2)) & 0x09249249;
			//	return n;
			//}

			//uint EncodeMorton(uint depthInt, uint bitDepth) {
			//	// e.g., bitDepth = 18 (6 bits per axis)
			//	uint bitsPerAxis = bitDepth / 3;
			//	uint mask = (1 << bitsPerAxis) - 1;

			//	uint x = (depthInt >> 0) & mask;
			//	uint y = (depthInt >> bitsPerAxis) & mask;
			//	uint z = (depthInt >> (2 * bitsPerAxis)) & mask;

			//	return (Part1By2(x) << 0) | (Part1By2(y) << 1) | (Part1By2(z) << 2);
			//}

			//float3 EncodeDepthToRGB(float depth, uint bitDepth) {
			//	uint maxDepth = (1 << bitDepth) - 1;
			//	uint depthInt = uint(depth * maxDepth);

			//	uint morton = EncodeMorton(depthInt, bitDepth);

			//	return float3(
			//		((morton >> 16) & 0xFF) / 255.0,
			//		((morton >> 8) & 0xFF) / 255.0,
			//		(morton & 0xFF) / 255.0
			//		);
			//}


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

			int DeinterleaveBits(int v)
			{
				v &= 0x49249;
				v = (v | (v >> 2)) & 0xC30C3;
				v = (v | (v >> 4)) & 0x0F00F;
				v = (v | (v >> 8)) & 0x000F;
				return v;
			}
			void DecodeMorton(int index, out int x, out int y, out int z)
			{
				int v = index;
				x = DeinterleaveBits(v);
				y = DeinterleaveBits(v >> 1);
				z = DeinterleaveBits(v >> 2);
			}

			int Compact1By2(int x)
			{
				x &= 0x492492;
				x = (x ^ (x >> 2)) & 0xC30C30C;
				x = (x ^ (x >> 4)) & 0x0F00F00F;
				x = (x ^ (x >> 8)) & 0x0000F00F;
				x = (x ^ (x >> 16)) & 0x0000000F;
				return x;
			}

			int DecodeMortonX(int morton) { return Compact1By2(morton); }
			int DecodeMortonY(int morton) { return Compact1By2(morton >> 1); }
			int DecodeMortonZ(int morton) { return Compact1By2(morton >> 2); }

			//Color Wrapping
			float3 EncodeDepthToColorWrapped(float depth) {

				
				float3 color = float3(0.0, 0.0, 0.0);

				if (depth >= 0.0 && depth <= (1.0 / 7.0)) 
				{
					color.r = (depth - (0.0 / 7.0)) / (1.0/7.0);
					color.g = 0.0;
					color.b = 0.0;
				}

				if (depth > (1.0 / 7.0) && depth <= (2.0 / 7.0))
				{
					color.r = 1.0;
					color.g = (depth - (1.0 / 7.0)) / (1.0/7.0);
					color.b = 0.0;
				}

				if (depth > (2.0 / 7.0) && depth <= (3.0 / 7.0))
				{
					color.r = 1.0-(depth - (2.0 / 7.0)) / (1.0 / 7.0);
					color.g = 1.0;
					color.b = 0.0;
				}

				if (depth > (3.0 / 7.0) && depth <= (4.0 / 7.0))
				{
					color.r = 0.0;
					color.g = 1.0;
					color.b = (depth - (3.0 / 7.0)) / (1.0 / 7.0);
				}

				if (depth > (4.0 / 7.0) && depth <= (5.0 / 7.0))
				{
					color.r = (depth - (4.0 / 7.0)) / (1.0 / 7.0);
					color.g = 1.0;
					color.b = 1.0;
				}

				if (depth > (5.0 / 7.0) && depth <= (6.0 / 7.0))
				{
					color.r = 1.0;
					color.g = 1.0-(depth - (5.0 / 7.0)) / (1.0 / 7.0);
					color.b = 1.0;
				}

				if (depth > (6.0 / 7.0) && depth <= (7.0 / 7.0))
				{
					color.r = 1.0 - (depth - (6.0 / 7.0)) / (1.0 / 7.0);
					color.g = 0.0;
					color.b = 1.0;
				}

				return color;
			}

			float3 YUVtoRGB(float3 yuv)
			{
				float Y = yuv.x;
				float U = yuv.y - 0.5;
				float V = yuv.z - 0.5;

				float R = Y + 1.5748 * V;
				float G = Y - 0.1873 * U - 0.4681 * V;
				float B = Y + 1.8556 * U;

				return float3(R, G, B);
			}
			float3 YUVtoRGB_BT709(float3 yuv) {
				float Y = yuv.x;
				float U = yuv.y - 0.5;
				float V = yuv.z - 0.5;

				float R = Y + 1.5748 * V;
				float G = Y - 0.1873 * U - 0.4681 * V;
				float B = Y + 1.8556 * U;

				return float3(R, G, B);
			}
			//float3 EncodeDepthToRGB(float depth, int yBits, int uvBits)
			//{
			//	// Clamp and quantize depth
			//	depth = saturate(depth);
			//	int totalBits = yBits + uvBits;
			//	int maxVal = (1 << totalBits) - 1;
			//	int intDepth = (int)(depth * maxVal + 0.5); // Round to nearest int

			//	// Split bits into Y and UV portions
			//	int yPart = intDepth >> uvBits; // Top yBits
			//	int uvPart = intDepth & ((1 << uvBits) - 1); // Bottom uvBits

			//	float Y = yPart / float((1 << yBits) - 1);
			//	float U = (uvPart >> (uvBits / 2)) / float((1 << (uvBits / 2)) - 1);
			//	float V = (uvPart & ((1 << (uvBits / 2)) - 1)) / float((1 << (uvBits / 2)) - 1);

			//	return YUVtoRGB_BT709(float3(Y, U, V)); // Packed into RGB space
			//}

			float3 EncodeDepthToRGB_YUVOrder(float depth)
			{
				// Clamp to [0,1] just in case
				depth = saturate(depth);

				// Split depth into 24 bits
				float d24 = depth * 16777215.0; // 2^24 - 1
				int dInt = (int)(d24 + 0.5);

				// Extract 8 bits at a time
				float high = (float)((dInt >> 16) & 0xFF) / 255.0;  // Most significant
				float mid = (float)((dInt >> 8) & 0xFF) / 255.0;
				float low = (float)(dInt & 0xFF) / 255.0;          // Least significant

				// Place high bits in Y, then U, then V
				float3 yuv = float3(high, mid, low);

				// Convert back to RGB so OBS can record it
				return YUVtoRGB_BT709(yuv);
			}

			//float3 EncodeDepthToRGB(float depth)
			//{
			//	depth = saturate(depth); // clamp between 0 and 1

			//	float r = floor(depth * 255.0) / 255.0;
			//	float g = floor(frac(depth * 255.0) * 255.0) / 255.0;
			//	float b = frac(depth * 255.0 * 255.0);

			//	return float3(r, g, b);
			//}
			float3 EncodeDepth16Bit(float depth)
			{
				depth = saturate(depth); // ensure depth is between 0 and 1

				uint value = (uint)(depth * 65535.0 + 0.5); // 16-bit integer value

				// Extract bits
				uint g = (value >> 8) & 0xFF;  // top 8 bits
				uint r = (value >> 4) & 0x0F;  // next 4 bits
				uint b = value & 0x0F;         // bottom 4 bits

				// Normalize to 0–1
				return float3(r / 15.0, g / 255.0, b / 15.0);
			}
			float3 EncodeDepth_RGBBucket(float depth)
			{
				// 1) clamp and scale to 15‑bit integer range [0..32767]
				depth = saturate(depth);
				int idx = (int)floor(depth * 32767.0 + 0.5);

				// 2) split into three 5‑bit indices: G = bits14–10, R = bits9–5, B = bits4–0
				int ig = (idx >> 10) & 31;
				int ir = (idx >> 5) & 31;
				int ib = idx & 31;

				// 3) map each 5‑bit index to the center of its 8‑value bin in [0..255]
				//    i.e. binSize = 256/32 = 8 → center at bin*8 + 4
				float fg = (ig * 8.0 + 4.0) / 255.0;
				float fr = (ir * 8.0 + 4.0) / 255.0;
				float fb = (ib * 8.0 + 4.0) / 255.0;

				return float3(fr, fg, fb);
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

			// -------- ENCODER --------
			float3 EncodeDepthToRGB(float depth) {
				depth = saturate(depth);

				// Clamp to avoid edge overflow
				uint maxCode = MAX_MORTON_CODE - 1;
				uint depthIndex = min((uint)(depth * maxCode + 0.5), maxCode);

				uint x, y, z;
				MortonDecode3D(depthIndex, x, y, z);

				float3 rgb;
				rgb.r = (x * BUCKET_SIZE + BUCKET_SIZE / 2) / 255.0;
				rgb.g = (y * BUCKET_SIZE + BUCKET_SIZE / 2) / 255.0;
				rgb.b = (z * BUCKET_SIZE + BUCKET_SIZE / 2) / 255.0;
				return rgb;
			}

			float3 EncodeDepthToRGB12bit(float depth) {
				// Clamp and scale depth to 12-bit integer (0–4095)
				depth = saturate(depth);
				int d = (int)(depth * 4095.0 + 0.5); // Round to nearest

				// Extract 4 bits per channel
				int g = (d >> 8) & 0xF;  // Top 4 bits
				int r = (d >> 4) & 0xF;  // Middle 4 bits
				int b = d & 0xF;         // Bottom 4 bits

				// Scale to 0–1 range for 8-bit texture channels
				return float3(r, g, b) / 15.0;
			}

			float3 EncodeDepthToRGB12M(float depth)
			{
				// 1) Quantize depth into [0 .. 4095]
				uint idx = (uint)floor(saturate(depth) * 4095.0 + 0.5);

				// 2) Decode the Morton index 'idx' into 4‑bit x,y,z coords in [0..15]
				uint x = 0, y = 0, z = 0;
				for (uint i = 0; i < 4; ++i)
				{
					uint bitMask = 1u << i;
					x |= ((idx >> (3 * i)) & 1u) << i;
					y |= ((idx >> (3 * i + 1)) & 1u) << i;
					z |= ((idx >> (3 * i + 2)) & 1u) << i;
				}

				// 3) Normalize to [0..1] for each channel
				return float3(x, y, z) / 15.0;
			}
			float3 EncodeDepthToRGB_12bit(float depth)
			{
				// clamp depth
				depth = saturate(depth);

				// quantize to [0..4095]
				float q = floor(depth * 4095.0 + 0.5);

				// split into two bytes
				uint depthInt = (uint)q;
				uint g = (depthInt >> 4) & 0xFF;      // upper 8 bits
				uint r = depthInt & 0x0F;             // lower 4 bits
				r = r << 4;                           // shift into MS half of R

				// convert to normalized limited‐range Rec709 (16–235)
				// optional: if you want to map 0→16 and 255→235
				float inv255 = 1.0 / 255.0;
				float low = 16.0 * inv255;
				float high = 235.0 * inv255;
				float3 outColor;
				outColor.r = lerp(low, high, r / 255.0);
				outColor.g = lerp(low, high, g / 255.0);
				outColor.b = low; // keep B at black

				return outColor;
			}

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

			// Encode depth ∈ [0,1] → RGB colormap
			float3 EncodeDepthToColor(float depth)
			{
				// optionally clamp: depth = saturate(depth);
				// map depth to hue range [0,0.8] to avoid red wrap seam
				float hue = depth * 0.8;
				// full saturation/value gives max contrast
				return HSVtoRGB(hue, 1.0, 1.0);
			}

			float3 LinearSrgbToOkLab(float3 c)
			{
				// linear sRGB → LMS
				float3 lms = float3(
					dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)),
					dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)),
					dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005))
					);
				// non‑linear transform
				lms = pow(lms, 1.0 / 3.0);
				// LMS → OKLab
				return float3(
					dot(lms, float3(0.2104542553, 0.7936177850, -0.0040720468)),
					dot(lms, float3(1.9779984951, -2.4285922050, 0.4505937099)),
					dot(lms, float3(0.0259040371, 0.7827717662, -0.8086757660))
					);
			}

			float3 OkLabToLinearSrgb(float3 lab)
			{
				// OKLab → LMS
				float3 lms = float3(
					dot(lab, float3(1.0000000, 0.3963377774, 0.2158037573)),
					dot(lab, float3(1.0000000, -0.1055613458, -0.0638541728)),
					dot(lab, float3(1.0000000, -0.0894841775, -1.2914855480))
					);
				// invert cubic root
				lms = lms * lms * lms;
				// LMS → linear sRGB
				return float3(
					dot(lms, float3(4.0767416621, -3.3077115913, 0.2309699292)),
					dot(lms, float3(-1.2684380046, 2.6097574011, -0.3413193965)),
					dot(lms, float3(-0.0041960863, -0.7034186147, 1.7076147010))
					);
			}
			float3 PackFloatToRGB24(float v)
			{
				// v scaled into [0,256³) space
				float scaled = v * 2.0; // use 16 bits—remap below
				float hi = floor(scaled / 256.0);
				float lo = frac(scaled / 256.0) * 256.0;
				// hi in [0,255], lo in [0,256)
				return float3(hi, lo, frac(scaled)) / 255.0;
			}

			// unpack back to float
			float UnpackRGB24ToFloat(float3 rgb)
			{
				rgb *= 255.0;
				float hi = rgb.r;
				float lo = rgb.g;
				float fra = rgb.b;
				// reconstruct the 16‑bit integer and normalize
				float reconstructed = hi * 256.0 + lo;
				return reconstructed / 65535.0;
			}
			float3 EncodeDepthOKLab(float depth)
			{
				// 1) pack → linear‐RGB
				float3 linRGB = PackFloatToRGB24(depth);
				// 2) linear‐RGB → OKLab
				float3 lab = LinearSrgbToOkLab(linRGB);
				// 3) OKLab → linear‐sRGB for output
				return OkLabToLinearSrgb(lab);
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

			float3 EncodeLoopHSV(float v)
			{
				// Scale into [0,6), pick segment 0…5
				float f = saturate(v) * 6.0;
				int seg = min(5, (int)floor(f));
				float t = f - seg;       // local frac ∈[0,1)

				// Each segment moves along one axis at a time:
				// 0: (1,0,0)->(1,1,0), 1: (1,1,0)->(0,1,0), 2: (0,1,0)->(0,1,1),
				// 3: (0,1,1)->(0,0,1), 4: (0,0,1)->(1,0,1), 5: (1,0,1)->(1,0,0)
				float3 c;
				if (seg == 0)      c = float3(1, t, 0);
				else if (seg == 1) c = float3(1 - t, 1, 0);
				else if (seg == 2) c = float3(0, 1, t);
				else if (seg == 3) c = float3(0, 1 - t, 1);
				else if (seg == 4) c = float3(t, 0, 1);
				else               c = float3(1, 0, 1 - t);

				return c;
			}
			float3 EncodeLoopHSV2(float v)
			{
				float p = v * 0.9;
				float3 color = (0,0,0);
				if (p > 0.0 && p <= (1.0/6.0)*0.9)
				{
					color = float3(1.0, p*(1.0/0.15)*0.9, 0.0);
				}
				else if (p > (1.0 / 6.0)*0.9 && p <= (2.0 / 6.0)*0.9)
				{
					color = float3(1.0 - ((p-0.15) * (1.0 / 0.15)*0.9), 1.0, 0.0);
				}
				else if (p > (2.0 / 6.0)*0.9 && p <= (3.0 / 6.0)*0.9)
				{
					color = float3(0.0, 1.0, ((p - 0.15*2.0) * (1.0 / 0.15)*0.9));
				}
				else if (p > (3.0 / 6.0)*0.9 && p <= (4.0 / 6.0)*0.9)
				{
					color = float3(0.0, 1.0 - ((p - 0.15*3.0) * (1.0 / 0.15)*0.9),1.0);
				}
				else if (p > (4.0 / 6.0)*0.9 && p <= (5.0 / 6.0)*0.9)
				{
					color = float3(((p - 0.15*4.0) * (1.0 / 0.15)*0.9), 0.0, 1.0);
				}
				else if (p > (5.0 / 6.0)*0.9 && p <= (6.0 / 6.0)*0.9)
				{
					color = float3(1.0, 0.0,1.0- ((p - 0.15*5.0) * (1.0 / 0.15)*0.9));
				}

				float f = saturate(v) * 6.0;
				int seg = min(5, (int)floor(f));

				float t = f - seg;
				t = lerp(0.0, 0.90, t); // shrink range into [0.05, 0.95]//come on gpt i told you 0 to .9

				float3 c;
				if (seg == 0)      c = float3(1, t, 0);		//(1,0,0) -> (1,1,0)
				else if (seg == 1) c = float3(1 - t, 1, 0);	//(1,1,0) -> (0,1,0)
				else if (seg == 2) c = float3(0, 1, t);		//(0,1,0) -> (0,1,1)
				else if (seg == 3) c = float3(0, 1 - t, 1);	//(0,1,1) -> (0,0,1)
				else if (seg == 4) c = float3(t, 0, 1);		//(0,0,1) -> (1,0,1)
				else               c = float3(1, 0, 1 - t);	//(1,0,1) -> (1,0,0)

				//return c;
				return color;
			}

			fixed4 frag(v2f i) : SV_Target
			{

				if (i.uv.y > 0.5)
				{
					//Shader motion uses a 6x45 grid of squares(tiles) the height of the screen, and a fraction of the width:82/1085 which seems to be close to .075<- value does round nicly with powers of 2 screen sizes. 
					//// Top half: show color texture
					//float2 uvRemap = i.uv * float2(1, 2) - float2(0, 1);
					//return tex2D(_TopTex, uvRemap);

					// -------- TOP HALF --------
					//float2 uvRemap = i.uv * float2(1, 2) - float2(0, 1); // scale UV from 0.5~1.0 to 0~1

					//// Get tile index for this pixel
					//uint2 tileXY = floor(uvRemap * tileCount);
					//tileXY.y = tileCount.y - 1 - tileXY.y; // flip Y axis to make origin top-left
					//uint tileIndex = (tileXY.x * tileCount.y + tileXY.y);

					//// Example: Only use tiles 0–3 for encoding data
					//if (tileIndex < 12)
					//{
					//	// Define 12 different float values
					//	float values[12] = {
					//		0.1, 0.2, 0.3,
					//		0.4, 0.5, 0.6,
					//		0.7, 0.8, 0.9,
					//		1.0, -0.5, -1.0
					//	};

					//	ColorTile tile;
					//	EncodeVideoSnorm(tile, values[tileIndex]);
					//	return RenderTile(tile, uvRemap);
					//}
					//else
					//{
					//	return tex2D(_TopTex, uvRemap); // passthrough color
					//}
					//float widthRatio =.075 ;

					//float2 uvRemap = i.uv * float2(1, 2) - float2(0, 1); // remap to 0–1

					////if (i.uv.x < widthRatio)
					////{
					////	// -------- TILE REGION --------
					////	// Normalize to tile space within the 0.075 screen width
					////	float2 localUV = float2(uvRemap.x / widthRatio, uvRemap.y);

					////	// Get tile coordinates in 6x45 grid
					////	uint2 tileXY = floor(localUV * tileCount);
					////	tileXY.y = tileCount.y - 1 - tileXY.y; // flip Y for top-down

					////	uint tileIndex = tileXY.x * tileCount.y + tileXY.y;

					////	if (tileIndex < 270)
					////	{
					////		float value = tileIndex / 269.0; // 0.0 to 1.0 gradient
					////		ColorTile tile;
					////		EncodeVideoSnorm(tile, value);
					////		return RenderTile(tile, localUV);
					////	}
					////}

					//// Outside tile strip: normal top texture
					//return tex2D(_TopTex, uvRemap);


					
					///*float height = 1.0;
					//float width= 0.075;
					//int gridx = 6;
					//int gridy = 45;*/

					////float2 uv = i.uv;
					//float2 uvRemap = i.uv * float2(1, 2) - float2(0, 1);
					//// Check if inside the top-left rectangle
					////if (uvRemap.x < width && uvRemap.y >(1.0 - height))
					//if (uvRemap.x > 0.0 && uvRemap.x < 0.025 && uvRemap.y >(1.0 - _Height))
					//{
					//	// Normalize coordinates within overlay region
					//	float2 localUV;
					//	localUV.x = uvRemap.x / _WidthRatio;
					//	localUV.y = (uvRemap.y - (1.0 - _Height)) / _Height;

					//	// Determine grid cell (0 to gridSize-1)
					//	int col = floor(localUV.x * _GridSize.x);
					//	int row = floor(localUV.y * _GridSize.y);

					//	// Flatten 2D cell index to 1D
					//	int index = row * int(_GridSize.x) + col;
					//	int total = int(_GridSize.x * _GridSize.y);

					//	//// Convert to grayscale 0 = white, 1 = black
					//	//float gray = 1.0 - (float(index) / (float(total) - 1.0));
					//	//return fixed4(gray, gray, gray, 1.0);
					//	//return float4(1, 1, 1, 1);
					//	if (index < _VisibleSquares)
					//	{
					//		float gray = 1.0 - (float(index) / max(float(90 - 1), 1.0));
					//		return fixed4(gray, gray, gray, 1.0);
					//	}
					//}
					//if (uvRemap.x > .025 && uvRemap.x < .050 && uvRemap.y >(1.0 - _Height))
					//{
					//	// Normalize coordinates within overlay region
					//	float2 localUV;
					//	localUV.x = uvRemap.x / _WidthRatio;
					//	localUV.y = (uvRemap.y - (1.0 - _Height)) / _Height;

					//	// Determine grid cell (0 to gridSize-1)
					//	int col = floor(localUV.x * _GridSize.x);
					//	int row = floor(localUV.y * _GridSize.y);

					//	// Flatten 2D cell index to 1D
					//	int index = row * int(_GridSize.x) + col;
					//	int total = int(_GridSize.x * _GridSize.y);

					//	//// Convert to grayscale 0 = white, 1 = black
					//	//float gray = 1.0 - (float(index) / (float(total) - 1.0));
					//	//return fixed4(gray, gray, gray, 1.0);
					//	//return float4(1, 1, 1, 1);
					//	if (index < _VisibleSquares)
					//	{
					//		float gray = 1.0 - (float(index) / max(float(90 - 1), 1.0));
					//		return fixed4(gray, gray, gray, 1.0);
					//	}
					//}

					// Otherwise, return the original texture color
					//return tex2D(_MainTex, uv);
					float2 uvRemap = i.uv * float2(1, 2) - float2(0, 1);
					//// Grid layout
					////float2 uv = i.uv;

					//// Parameters
					//const int cols = 3;
					//const int rows = 45;
					//const float slotWidth = 0.025/2.0;
					//const float slotHeight = 1.0 / rows;

					//// Get column and row for this UV
					//float2 localUV = uvRemap;

					//// Determine which slot this UV would fall into
					//int col = (int)(localUV.x / (2.0 * slotWidth));
					//int row = (int)((1.0 - localUV.y) / slotHeight);

					//// Compute slot index in top-to-bottom, left-to-right order
					//int slotIndex = col * rows + row;

					//// UV within the slot
					//float2 inSlotUV = float2(fmod(localUV.x, 2.0 * slotWidth), fmod(localUV.y, slotHeight));
					//bool inSquare = inSlotUV.x < slotWidth * 2;
					//bool inLeftTile = inSlotUV.x < slotWidth;

					//// Show only if within visible count and inside square region
					//if (slotIndex < _VisibleSlotCount && inSquare)
					//{
					//	if (slotIndex == 0 )
					//		return float4(0, 1, 0, 1); // Left tile of slot 0 is green

					//	return float4(1, 0, 0, 1); // All other tiles are red
					//}
					return tex2D(_TopTex, uvRemap);


				}
				else
				{
					//float near = _Near;
					//float far = _Far;
					//float gamma = _Gamma;
					// Bottom half: show depth as grayscale
					//float2 uvRemap = i.uv * float2(1, 2);
					//float depth = pow(1.0-(tex2D(_BottomDepthTex, uvRemap).r,2.2));
					//float depth = pow(1.0 - tex2D(_BottomDepthTex, uvRemap).r, gamma);
					//float linearDepth = Linear01Depth(depth, near, far);
					//float linearDepth =  LinearEyeDepth(depth,near,far);
					//float depthLog = log(1.0 + linearDepth); // Add 1.0 to avoid log(0)
					float gamma = _Gamma;
					float2 uvRemap = i.uv * float2(1, 2);
					//float depth = pow(1.0 - tex2D(_BottomDepthTex, uvRemap).r, gamma);//revsered the unity reversed 
					float depth = pow(tex2D(_BottomDepthTex, uvRemap).r, gamma);
					//perspective depth is nonlinear, so convert it to linear before encoding??
					//depth = LinearToGammaDepth(depth);

					////NAIVE COLOR ENCODE
					//float r = saturate((depth - 0.0) / 0.33333);
					//float g = saturate((depth - 0.33333) / 0.33333);
					//float b = saturate((depth - 0.66666) / 0.33334); // use 0.33334 for precision
					//return float4(r, g, b, 1);
					
					//NAIVE COLOR ENCODE with lmited color ranging.//wait, i think the video encoding already kinda deals with this, no need to do it within the shader
					//float r = saturate((depth - 0.0) / 0.33333);
					//r = lerp(16.0 / 255.0, 235.0 / 255.0, r);
					//float g = saturate((depth - 0.33333) / 0.33333);
					//g = lerp(16.0 / 255.0, 235.0 / 255.0, g);
					//float b = saturate((depth - 0.66666) / 0.33334); // use 0.33334 for precision
					//b = lerp(16.0 / 255.0, 235.0 / 255.0, b);
					//return float4(r, g, b, 1);

					////Morton 
					//int morton = (int)(depth * 4096); // MAX_MORTON should be your highest used index, e.g. 4096^3

					//// Decode Morton index into x, y, z
					//int x = DecodeMortonX(morton);
					//int y = DecodeMortonY(morton);
					//int z = DecodeMortonZ(morton);

					//// Optional: Normalize coordinates (e.g., to [0,1] if max dimension known)
					//float3 coord = float3(x, y, z) / float3(16, 16, 16); // Replace with your dimensions

					//// Return as color
					//return float4(coord, 1.0);

					//Wrapping around the color space more
					//float r = saturate((depth - 0.0) / 0.33333);
					//float g = saturate((depth - 0.33333) / 0.33333);
					//float b = saturate((depth - 0.66666) / 0.33334); // use 0.33334 for precision

					////COlor Wrapping (Jet)
					//float3 color = EncodeDepthToColorWrapped(depth);
					//return float4(color, 1);

					////24Bit RGB Depth Encoding
					//uint depthInt = (uint)(depth * 16777215.0); // 2^24 - 1

					//float r = (depthInt >> 16) & 0xFF;
					//float g = (depthInt >> 8) & 0xFF;
					//float b = depthInt & 0xFF;

					//float3 color = float3(r, g, b) / 255.0; // Normalize to 0-1 for output

					//return float4(color, 1);
					//24bit encode
					// Input: depth in [0, 1]
					//float depthClamped = saturate(depth); // Ensure it's clamped
					//uint depthInt = (uint)(depthClamped * 16777215.0); // 2^24 - 1

					//float r = (depthInt >> 16) & 0xFF;
					//float g = (depthInt >> 8) & 0xFF;
					//float b = depthInt & 0xFF;

					//float3 color = float3(r, g, b) / 255.0; // Normalize to 0-1 for output
					//return float4(color, 1.0);

					////YUV 24bit Ordered encode
					//// Clamp depth and convert to 24-bit int
					////float depth = saturate(_Depth);
					//uint dInt = (uint)(depth * 16777215.0 + 0.5);

					//// MSBs in Y (16 bits), LSBs in U/V (8 bits)
					//uint msb = (dInt >> 8) & 0xFFFF;
					//uint lsb = dInt & 0xFF;

					//float y = msb / 65535.0;
					//float u = ((lsb >> 4) / 15.0) * 0.5 + 0.25; // Upper nibble
					//float v = ((lsb & 0xF) / 15.0) * 0.5 + 0.25; // Lower nibble

					//float3 rgb = YUVtoRGB(y, u, v);
					//return float4(rgb, 1.0);

					//YUV dynamic 24-n bit depth encode
					//// Clamp depth and convert to 24-bit int
					////float depth = saturate(_Depth);
					////uint dInt = (uint)(depth * 16777215.0 + 0.5);

					//uint totalBits = 24;
					//uint msbBits = clamp(_DepthBitsInY, 8, 24);
					//uint lsbBits = totalBits - msbBits;

					//uint dInt = (uint)(saturate(depth) * ((1u << totalBits) - 1));

					//int msb = dInt >> lsbBits;
					//uint lsb = dInt & ((1u << lsbBits) - 1);

					//// Normalize based on bit count
					//float y = msb / (float)((1u << msbBits) - 1);

					//// Use UV to store remaining bits
					//float u = 0.5, v = 0.5;
					//if (lsbBits > 0)
					//{
					//	uint lsbHigh = lsb >> (lsbBits / 2);
					//	uint lsbLow = lsb & ((1u << (lsbBits / 2)) - 1);

					//	u = (lsbHigh / (float)((1u << (lsbBits / 2)) - 1)) * 0.5 + 0.25;
					//	v = (lsbLow / (float)((1u << (lsbBits / 2)) - 1)) * 0.5 + 0.25;
					//}
					////float3 comYUV = (y, u, v);
					//float3 rgb = YUVtoRGB((y, u, v));
					//return float4(rgb, 1.0);

					//slop
					//return float4(EncodeDepthToRGB(depth, 8, 8), 1.0);
					//return float4(EncodeDepthToRGB_YUVOrder(depth), 1.0);
					//return float4(EncodeDepthToRGB(depth), 1.0);
					//return float4(EncodeDepth16Bit(depth), 1.0);
					//return float4(EncodeDepthToColorWrapped(depth), 1.0);
					//return float4(EncodeDepthMorton(depth), 1.0);
					//return float4(EncodeDepthToRGB(depth), 1.0);
					//return float4(EncodeDepthToRGB12bit(depth), 1.0);
					//return float4(EncodeDepthToRGB12M(depth), 1.0);
					//return float4(EncodeDepthToColor(depth), 1.0);
					//return float4(EncodeDepthOKLab(depth), 1.0);
					//return float4(EncodeDepthToRGB_Oklab(depth), 1.0);
					//return float4(EncodeDepthToColorWrapped(depth), 1.0);
					//return float4(EncodeLoopHSV(depth), 1.0);
					//return float4(EncodeDepth_RGBBucket(depth), 1.0);
					return float4(EncodeLoopHSV2(depth), 1.0);

					//return float4(depth, depth, depth, 1); 
					//depth = log(depth);
					// float4(depth, depth, depth, 1); // Grayscale
				}
			}
			ENDCG
		}
	}
}
