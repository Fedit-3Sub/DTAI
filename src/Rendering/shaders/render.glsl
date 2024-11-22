precision mediump float;

uniform sampler2D u_material;
uniform vec2 u_obstaclePosition;
uniform vec2 u_textureSize;
uniform float u_obstacleRad;

// 빛나는 효과를 부드럽게 만들기 위한 함수
float softGlow(vec2 fragCoord, vec2 center, float radius, float intensity) {
    float dist = length(fragCoord - center);
    return pow(exp(-dist / radius), intensity);
}

// 더욱 생동감 있는 색상 매핑 함수
vec3 aestheticWindColor(float intensity) {
    vec3 lightBlue = vec3(0.7, 0.85, 1.0);    // 연한 하늘색
    vec3 teal = vec3(0.2, 0.8, 0.9);          // 청록색
    vec3 deepBlue = vec3(0.1, 0.3, 0.8);      // 깊은 파란색
    vec3 glowingWhite = vec3(1.2, 0.8, 1); // 빛나는 흰색-황색 톤
    

    if (intensity < 0.15) {
        return mix(vec3(0.9, 0.95, 1.0), lightBlue, intensity * 4.0);
    } else if (intensity < 0.5) {
        return mix(lightBlue, teal, (intensity - 0.25) * 4.0);
    } else if (intensity < 0.75) {
        return mix(teal, deepBlue, (intensity - 0.5) * 4.0);
    }
    return mix(deepBlue, glowingWhite, (intensity - 0.75) * 4.0);
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;

    // 장애물 처리
    vec2 dir = fragCoord - vec2(0.5, 0.5) - u_obstaclePosition;
    float dist = length(dir);
    if (dist < u_obstacleRad) {
        gl_FragColor = vec4(0.05, 0.1, 0.3, 1.0); // 장애물은 어두운 남색
        return;
    }

    // 유체 흐름 데이터 처리
    float mat1 = texture2D(u_material, fragCoord / u_textureSize).x;

    // 색상 계산
    vec3 flowColor = aestheticWindColor(mat1);

    // 빛나는 효과 추가
    float glow = softGlow(fragCoord, vec2(0.5, 0.5) + u_obstaclePosition, u_obstacleRad * 2.0, 1.5);
    vec3 glowColor = vec3(1.2, 1.0, 0.8); // 황금빛으로 빛나는 색상

    // 색상과 빛 효과를 조화롭게 합성
    vec3 finalColor = mix(flowColor, glowColor, glow * 0.5);

    // 흐름의 부드러움을 강조하는 알파값
    float alpha = mat1 * 0.9 + glow * 0.15;

    gl_FragColor = vec4(finalColor, alpha);
}
