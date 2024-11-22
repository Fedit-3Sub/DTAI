precision mediump float;

uniform sampler2D u_material;
uniform vec2 u_obstaclePosition;
uniform vec2 u_textureSize;
uniform float u_obstacleRad;

vec3 windColor(float intensity) {
    // 기본 색상 정의
    vec3 lightWind = vec3(0.85, 0.95, 1.0);   // 매우 연한 하늘색
    vec3 mediumWind = vec3(0.6, 0.8, 0.95);   // 중간 하늘색
    vec3 strongWind = vec3(0.4, 0.6, 0.9);    // 진한 하늘색
    vec3 veryStrongWind = vec3(0.2, 0.4, 0.8); // 매우 진한 하늘색
    
    // 그라데이션 효과 생성
    if (intensity < 0.25) {
        return mix(vec3(0.95, 0.98, 1.0), lightWind, intensity * 4.0);
    } else if (intensity < 0.5) {
        return mix(lightWind, mediumWind, (intensity - 0.25) * 4.0);
    } else if (intensity < 0.75) {
        return mix(mediumWind, strongWind, (intensity - 0.5) * 4.0);
    }
    return mix(strongWind, veryStrongWind, (intensity - 0.75) * 4.0);
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    
    // 장애물 처리
    vec2 dir = fragCoord - vec2(0.5, 0.5) - u_obstaclePosition;
    float dist = length(dir);
    if (dist < u_obstacleRad) {
        // 장애물은 더 진한 남색으로 표현
        gl_FragColor = vec4(0.05, 0.1, 0.25, 1.0);
        return;
    }
    
    // 유체 흐름 처리
    float mat1 = texture2D(u_material, fragCoord/u_textureSize).x;
    
    // 바람 효과 적용
    vec3 flowColor = windColor(mat1);
    
    // 알파값을 높여서 더 진하게 표현
    float alpha = mat1 * 0.95 + 0.2;  // 알파값 약간 증가

    gl_FragColor = vec4(flowColor, alpha);
}