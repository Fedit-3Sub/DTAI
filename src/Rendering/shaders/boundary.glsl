precision mediump float;

uniform sampler2D u_texture;
uniform float u_scale;
uniform vec2 u_textureSize;

uniform vec2 u_obstaclePosition;//scaled to texture size
uniform float u_obstacleRad;

void main() {
    vec2 fragCoord = gl_FragCoord.xy;

    vec2 dir = fragCoord - 3.0*vec2(0.5, 0.5) - u_obstaclePosition;
    float dist = length(dir);
    
    if (dist < u_obstacleRad){
        gl_FragColor = vec4(0);
        return;
    }

    if (dist < u_obstacleRad+1.0){
        gl_FragColor = u_scale*texture2D(u_texture, (fragCoord + dir/dist*2.0)/u_textureSize);
        return;
    }

    gl_FragColor = texture2D(u_texture, fragCoord/u_textureSize);
}