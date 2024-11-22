uniform float u_initialVelocityX;

void main() {
    vec2 coord = gl_FragCoord.xy;
    if (coord.x == 0.0) {
        gl_FragColor = vec4(u_initialVelocityX, 0.0, 0.0, 0.0);
    } else {
        gl_FragColor = vec4(0.0);
    }
}