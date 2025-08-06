Shader "Custom/EncodeData"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}

		_VisibleSlotCount("Visible Slots", Range(0,135)) = 135

		_PosX("X", Float) = 0.0
		_PosY("Y", Float) = 0.0
		_PosZ("Z", Float) = 0.0

		_QuaterionX("Rotation Quaterion", Float) = 0.0
		_QuaterionY("Rotation Quaterion", Float) = 0.0
		_QuaterionZ("Rotation Quaterion", Float) = 0.0
		_QuaterionW("Rotation Quaterion", Float) = 0.0
		/*_RotY("Transform local Y", Vector) = (0.0, 0.0, 0.0)
		_RotZ("Transform local Z", Vector) = (0.0, 0.0, 0.0)*/

		_FOVSIZE("FOV/SIZE", Float) = 0.0

		_NEAR("Near Plane", Float) = 0.0
		_FAR("Far Plane", Float) = 0.0
		[ToggleUI] _isOrtho("Is Ortho?",Float)=1.0
		//_FOVSIZEY("FOV/SIZE", Float) = 0.0
		//_FOVSIZEZ("FOV/SIZE", Float) = 0.0
		//_Scale("Scale", Float) = 1.0
    }
    SubShader
    {
		Tags { "QUEUE" = "Transparent" "IGNOREPROJECTOR" = "true" "RenderType" = "Transparent" }
		//Tags { "RenderType" = "Opaque" }
		ZWrite Off
		Blend SrcAlpha OneMinusSrcAlpha
		Cull back
		LOD 100
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
			#include "Codec.hlsl"
			#include "VideoLayout.hlsl"
			float _PosX;
			float _PosY;
			float _PosZ;
			float _QuaterionX;
			float _QuaterionY;
			float _QuaterionZ;
			float _QuaterionW;
			//float3 _RotY;
			//float3 _RotZ;
			//float _Scale;
			int _VisibleSlotCount;
			float _FOVSIZE;
			float _NEAR;
			float _FAR;
			float _isOrtho;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };
			struct FragInputTile {
				nointerpolation ColorTile color : COLOR;
				float2 uv : TEXCOORD0;
				float4 pos : SV_Position;
				UNITY_VERTEX_OUTPUT_STEREO
			};

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }
			float4 LinearToGamma(float4 linearColor) {
				return pow(linearColor, 2.2);
			}
            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                //fixed4 col = tex2D(_MainTex, i.uv);
                // apply fog
                //UNITY_APPLY_FOG(i.fogCoord, col);

				const int cols = 3;
				const int rows = 45;
				const float slotWidth = 0.16667;
				const float slotHeight = 1.0 / rows;

				float2 uvRemap = i.uv;
				float2 localUV = uvRemap;

				// Determine which slot this UV would fall into
				int col = (int)(localUV.x / (2.0 * slotWidth));
				int row = (int)((1.0 - localUV.y) / slotHeight);

				// Compute slot index in top-to-bottom, left-to-right order
				int slotIndex = col * rows + row;

				// UV within the slot
				float2 inSlotUV = float2(fmod(localUV.x, 2.0 * slotWidth), fmod(localUV.y, slotHeight));
				bool inSquare = inSlotUV.x < slotWidth * 2;
				bool inLeftTile = inSlotUV.x < slotWidth;

				// Show only if within visible count and inside square region
				if (slotIndex < _VisibleSlotCount && inSquare)
				{
					//Low
					if (slotIndex == 0)//0
					{
						ColorTile encoded;
						float data = _PosX/2.0;
						if (inLeftTile) 
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							return LinearToGamma(guh);
						}
						else 
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 1)
					{
						ColorTile encoded;
						float data = _PosY / 2.0;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 2)
					{
						ColorTile encoded;
						float data = _PosZ / 2.0;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							return LinearToGamma(guh);
						}
					}
					//High
					if (slotIndex == 3)
					{
						ColorTile encoded;
						float data = _PosX / 2.0;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 4)
					{
						ColorTile encoded;
						float data = _PosY / 2.0;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 5)
					{
						ColorTile encoded;
						float data = _PosZ / 2.0;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							return LinearToGamma(guh);
						}
					}
					//scale???
					if (slotIndex == 6)
					{
						ColorTile encoded;
						float data = _QuaterionX;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 7)
					{
						ColorTile encoded;
						float data = _QuaterionY;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 0.5, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(0.83984, 0.0, 0.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 8)
					{
						ColorTile encoded;
						float data = _QuaterionZ;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
							return LinearToGamma(guh);

						}
					}
					if (slotIndex == 9)
					{
						ColorTile encoded;
						float data = _QuaterionW;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
							return LinearToGamma(guh);
						}
					}
					//if (slotIndex == 10)
					//{
					//	ColorTile encoded;
					//	float data = _RotZ.y;
					//	if (inLeftTile)
					//	{
					//		EncodeVideoSnorm(encoded, data, false);
					//		float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
					//		//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
					//		return LinearToGamma(guh);
					//	}
					//	else
					//	{
					//		EncodeVideoSnorm(encoded, data, false);
					//		float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
					//		//float4 guh = float4(0.5, 0.5, 0.5, 1.0);
					//		return LinearToGamma(guh);
					//	}
					//}
					//if (slotIndex == 11)
					//{
					//	ColorTile encoded;
					//	float data = _RotZ.z;
					//	if (inLeftTile)
					//	{
					//		EncodeVideoSnorm(encoded, data, false);
					//		float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
					//		//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
					//		return LinearToGamma(guh);
					//	}
					//	else
					//	{
					//		EncodeVideoSnorm(encoded, data, false);
					//		float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
					//		//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
					//		return LinearToGamma(guh);
					//	}
					//}
					//swingtwist test
					if (slotIndex == 10)
					{
						ColorTile encoded;
						float data = _FOVSIZE;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 11)
					{
						ColorTile encoded;
						float data = _FOVSIZE;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					//Near
					if (slotIndex == 12)
					{
						ColorTile encoded;
						float data = _NEAR;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 13)
					{
						ColorTile encoded;
						float data = _NEAR;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					//Far
					if (slotIndex == 14)
					{
						ColorTile encoded;
						float data = _FAR;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, true);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					if (slotIndex == 15)
					{
						ColorTile encoded;
						float data = _FAR;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}
					//ortho
					if (slotIndex == 16)
					{
						ColorTile encoded;
						float data = _isOrtho;
						if (inLeftTile)
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[0].r, encoded[0].g, encoded[0].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
						else
						{
							EncodeVideoSnorm(encoded, data, false);
							float4 guh = float4(encoded[1].r, encoded[1].g, encoded[1].b, 1.0);
							//float4 guh = float4(1.0, 1.0, 1.0, 1.0);
							return LinearToGamma(guh);
						}
					}



						

					return LinearToGamma(float4(.5, 0.5, 0.5, 1));
				}
				return LinearToGamma(float4(0.0, 0.0, 0.0, 0.0));
				/*float4 col = float4(0.0,0.0,0.0,1.0);

                return col;*/
            }
            ENDCG
        }
    }
}
