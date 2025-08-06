Shader "Custom/DepthToGrayscale"
{
	SubShader
	{
		Tags { "RenderType" = "Opaque" }
		Pass
		{
			ZWrite On
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#include "UnityCG.cginc"

			struct v2f
			{
				float4 pos : SV_POSITION;
				float depth : TEXCOORD0;
			};

			v2f vert(appdata_base v)
			{
				v2f o;
				float4 worldPos = mul(unity_ObjectToWorld, v.vertex);
				o.pos = UnityObjectToClipPos(v.vertex);
				o.depth = o.pos.z / o.pos.w; // Clip space depth
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				// Convert clip space depth to linear eye depth (0–1)
				float linearDepth = Linear01Depth(i.depth);
				return fixed4(linearDepth, linearDepth, linearDepth, 1.0);
			}
			ENDCG
		}
	}
		Fallback Off
}