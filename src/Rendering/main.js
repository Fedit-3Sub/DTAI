// 시뮬레이션 영역 관련 설정
const scale = 3.5; // 스케일값
let width, height; // 시뮬레이션 너비와 높이
let actualWidth, actualHeight; // 실제 화면 너비와 높이
let paused = false; // 창 크기 조정 중 일시 정지 상태

// 장애물 관련 설정
let obstacleRad = 30; // 장애물 반지름
let obstaclePosition = [0,0]; // 장애물 위치
let movingObstacle = false; // 장애물 움직이는지 여부
let obstacleEnabled = true; // 장애물 활성화 상태 (기본값: 활성화)


// 마우스 관련 설정
let lastMouseCoordinates =  [0,0]; // 마지막 마우스 좌표
let mouseCoordinates =  [0,0]; // 현재 마우스 좌표
let mouseEnable = false; // 마우스 입력 활성화 여부 

// 시뮬레이션 관련 설정
const dx = 1; // 공간 간격
let dt = 1; // 시간 간격
let mu = 1; // 점성 계수
let rho = 1; // 밀도

var GPU;
window.onload = initGL; // 페이지 로드 시 initGL 함수 호출
window.updateProperties = updateProperties;


// 특성값 업데이트
function updateProperties(values) {
    if (values.rho !== undefined) {
        rho = Number(values.rho);
        resetWindow();
    }
    if (values.obstacleRad !== undefined) {
        obstacleRad = Number(values.obstacleRad);
        resetWindow();
    }
    if (values.mu !== undefined) {
        mu = Number(values.mu);
        resetWindow();
    }
    if (values.dt !== undefined) {
        dt = Number(values.dt);
        GPU.setProgram("advectVel", "2d-vertex-shader", "advectShaderVel");
        GPU.setUniformForProgram("advectVel", "u_dt", dt, "1f");
        GPU.setProgram("advectMat", "2d-vertex-shader", "advectShaderMat");
        GPU.setUniformForProgram("advectMat", "u_dt", dt, "1f");
        GPU.setProgram("gradientSubtraction", "2d-vertex-shader", "gradientSubtractionShader");
        GPU.setUniformForProgram("gradientSubtraction", "u_const", dt/(2*rho*dx), "1f");
        GPU.setProgram("diverge", "2d-vertex-shader", "divergenceShader");
        GPU.setUniformForProgram("diverge", "u_const", dt/(2*rho*dx), "1f");
        GPU.setProgram("force", "2d-vertex-shader", "forceShader");
        GPU.setUniformForProgram("force", "u_dt", dt, "1f");
        resetWindow();
    }
    
}

// 쉐이더 로드
async function loadShader(url) {
    const response = await fetch(url);
    const shaderSource = await response.text();
    return shaderSource;
}

// 각종 쉐이더 초기화
async function initShaders() {
    const advectVelShaderSource = await loadShader('shaders/advectShaderVel.glsl');
    const advectVelshaderScript = document.createElement('script');
    advectVelshaderScript.id = 'advectShaderVel';
    advectVelshaderScript.type = 'x-shader/x-fragment';
    advectVelshaderScript.textContent = advectVelShaderSource;
    document.head.appendChild(advectVelshaderScript);

    const advectMatShaderSource = await loadShader('shaders/advectMat.glsl');
    const advectMatshaderScript = document.createElement('script');
    advectMatshaderScript.id = 'advectShaderMat';
    advectMatshaderScript.type = 'x-shader/x-fragment';
    advectMatshaderScript.textContent = advectMatShaderSource;
    document.head.appendChild(advectMatshaderScript);

    const jacobiShaderSource = await loadShader('shaders/jacobi.glsl');
    const jacobiShaderScript = document.createElement('script');
    jacobiShaderScript.id = 'jacobiShader';
    jacobiShaderScript.type = 'x-shader/x-fragment';
    jacobiShaderScript.textContent = jacobiShaderSource;
    document.head.appendChild(jacobiShaderScript);

    const forceShaderSource = await loadShader('shaders/force.glsl');
    const forceShaderScript = document.createElement('script');
    forceShaderScript.id = 'forceShader';
    forceShaderScript.type = 'x-shader/x-fragment';
    forceShaderScript.textContent = forceShaderSource;
    document.head.appendChild(forceShaderScript);

    const divergeShaderSource = await loadShader('shaders/diverge.glsl');
    const divergeShaderScript = document.createElement('script');
    divergeShaderScript.id = 'divergenceShader';
    divergeShaderScript.type = 'x-shader/x-fragment';
    divergeShaderScript.textContent = divergeShaderSource;
    document.head.appendChild(divergeShaderScript);   

    const gradientSubtractionShaderSource = await loadShader('shaders/gradientSubtraction.glsl');
    const gradientSubtractionShaderScript = document.createElement('script');
    gradientSubtractionShaderScript.id = 'gradientSubtractionShader';
    gradientSubtractionShaderScript.type = 'x-shader/x-fragment';
    gradientSubtractionShaderScript.textContent = gradientSubtractionShaderSource;
    document.head.appendChild(gradientSubtractionShaderScript);   

    const boundaryShaderSource = await loadShader('shaders/boundary.glsl');
    const boundaryShaderScript = document.createElement('script');
    boundaryShaderScript.id = 'boundaryConditionsShader';
    boundaryShaderScript.type = 'x-shader/x-fragment';
    boundaryShaderScript.textContent = boundaryShaderSource;
    document.head.appendChild(boundaryShaderScript);   

    const resetVelocityShaderSource = await loadShader('shaders/resetVelocity.glsl');
    const resetVelocityShaderScript = document.createElement('script');
    resetVelocityShaderScript.id = 'resetVelocityShader';
    resetVelocityShaderScript.type = 'x-shader/x-fragment';
    resetVelocityShaderScript.textContent = resetVelocityShaderSource;
    document.head.appendChild(resetVelocityShaderScript);   

    const renderShaderSource = await loadShader('shaders/render.glsl');
    const renderShaderScript = document.createElement('script');
    renderShaderScript.id = '2d-render-shader';
    renderShaderScript.type = 'x-shader/x-fragment';
    renderShaderScript.textContent = renderShaderSource;
    document.head.appendChild(renderShaderScript);   
}

async function initGL() {
    canvas = document.getElementById("glcanvas"); // 캔버스 요소 가져오기

    // 장애물 활성화/비활성화 버튼 클릭 이벤트 핸들러 설정
    document.getElementById('toggleObstacle').addEventListener('click', toggleObstacle);

    // canvas에 마우스 및 터치 이벤트 핸들러 설정
    canvas.onmousemove = onMouseMove;
    canvas.ontouchmove = onTouchMove;
    canvas.onmousedown = onMouseDown;
    canvas.onmouseup = onMouseUp;

    // 창 크기 조정 시 onResize 함수 호출
    window.onresize = onResize;

    // 셰이더 초기화
    await initShaders();

    // GPU 연산 초기화
    GPU = initGPUMath();

    // OpenGL Shading Language programs 셋업
    // advectVel : 프로그램 이름 설정
    // 2d-vertex-shader : vertex shader의 소스코드/ID
    // advectShaderVel : 프래그먼트 셰이더의 소스코드/ID
    GPU.createProgram("advectVel", "2d-vertex-shader", "advectShaderVel");
    GPU.setUniformForProgram("advectVel", "u_dt", dt, "1f"); // u_dt(변수), dt(값), 1f(타입,단일 부동소수점)
    GPU.setUniformForProgram("advectVel", "u_velocity", 0, "1i"); // u_velocity(변수), 0(값), 1i(타입,단일 정수)
    GPU.setUniformForProgram("advectVel", "u_material", 1, "1i"); // u_material(변수), 1(값), 1i(타입,단일 정수)

    // advectMat
    GPU.createProgram("advectMat", "2d-vertex-shader", "advectShaderMat");
    GPU.setUniformForProgram("advectMat", "u_dt", dt, "1f");
    GPU.setUniformForProgram("advectMat", "u_velocity", 0, "1i");
    GPU.setUniformForProgram("advectMat", "u_material", 1, "1i");

    // gradientSubtraction
    GPU.createProgram("gradientSubtraction", "2d-vertex-shader", "gradientSubtractionShader");
    GPU.setUniformForProgram("gradientSubtraction", "u_const", dt/(2*rho*dx), "1f");
    GPU.setUniformForProgram("gradientSubtraction", "u_velocity", 0, "1i");
    GPU.setUniformForProgram("gradientSubtraction", "u_pressure", 1, "1i");

    // diverge
    GPU.createProgram("diverge", "2d-vertex-shader", "divergenceShader");
    GPU.setUniformForProgram("diverge", "u_const", dt/(2*rho*dx), "1f");
    GPU.setUniformForProgram("diverge", "u_velocity", 0, "1i");

    // force
    GPU.createProgram("force", "2d-vertex-shader", "forceShader");
    GPU.setUniformForProgram("force", "u_dt", dt, "1f");
    GPU.setUniformForProgram("force", "u_velocity", 0, "1i");

    // jacobi
    GPU.createProgram("jacobi", "2d-vertex-shader", "jacobiShader");
    GPU.setUniformForProgram("jacobi", "u_b", 0, "1i");
    GPU.setUniformForProgram("jacobi", "u_x", 1, "1i");

    // render
    GPU.createProgram("render", "2d-vertex-shader", "2d-render-shader");
    GPU.setUniformForProgram("render", "u_material", 0, "1i");

    // boundary
    GPU.createProgram("boundary", "2d-vertex-shader", "boundaryConditionsShader");
    GPU.setUniformForProgram("boundary", "u_texture", 0, "1i");

    // resetVelocity
    GPU.createProgram("resetVelocity", "2d-vertex-shader", "resetVelocityShader");

    // 초기화
    resetWindow();

    // 렌더링 수행
    render();
}

// 창 화면 조절시 paused 값 true 로 지정
// render 함수에서 resetWindow 함수로 넘어감.
function onResize(){
    paused = true;
}

function resetWindow(){
    // 시뮬레이션 영역 너비 & 높이 
    actualWidth = canvas.clientWidth;
    actualHeight = canvas.clientHeight;

    // 스케일값 지정 (시뮬레이션의 해상도를 조절하는 역할)
    // -> 값이 클수록 작은 영역에 대해 유체 해석함. GPU, CPU 사용량 감소
    var maxDim = Math.max(actualHeight, actualWidth);
    var scale = maxDim/200;

    width = Math.floor(actualWidth/scale);
    height = Math.floor(actualHeight/scale);

    // 장애물 초기 위치 (너비의 1/10, 높이의 1/2)
    obstaclePosition = [actualWidth/5, actualHeight/2];

    // 드로잉 버퍼 사이즈를 실제 값과 맞춤
    canvas.width = actualWidth;
    canvas.height = actualHeight;

    GPU.setSize(width, height);

    // advectVel
    GPU.setProgram("advectVel");
    GPU.setUniformForProgram("advectVel" ,"u_textureSize", [width, height], "2f");
    GPU.setUniformForProgram("advectVel" ,"u_scale", 1, "1f");

    // advectMat
    GPU.setProgram("advectMat");
    GPU.setUniformForProgram("advectMat" ,"u_textureSize", [actualWidth, actualHeight], "2f");
    GPU.setUniformForProgram("advectMat" ,"u_scale", width/actualWidth, "1f");

    // gradientSubtraction
    GPU.setProgram("gradientSubtraction");
    GPU.setUniformForProgram("gradientSubtraction", "u_velocity", 0, "1i");
    GPU.setUniformForProgram("gradientSubtraction", "u_pressure", 1, "1i");
    GPU.setUniformForProgram("gradientSubtraction" ,"u_textureSize", [width, height], "2f");

    // diverge
    GPU.setProgram("diverge");
    GPU.setUniformForProgram("diverge", "u_const", dt/(2*rho*dx), "1f");
    GPU.setUniformForProgram("diverge", "u_velocity", 0, "1i");
    GPU.setUniformForProgram("diverge" ,"u_textureSize", [width, height], "2f");

    // force
    GPU.setProgram("force");
    GPU.setUniformForProgram("force", "u_reciprocalRadius", 0.1*scale, "1f");
    GPU.setUniformForProgram("force" ,"u_textureSize", [width, height], "2f");

    // jacobi
    GPU.setProgram("jacobi");
    GPU.setUniformForProgram("jacobi" ,"u_textureSize", [width, height], "2f");

    // render
    GPU.setProgram("render");
    GPU.setUniformForProgram("render", "u_obstaclePosition", [obstaclePosition[0], obstaclePosition[1]], "2f");
    GPU.setUniformForProgram("render", "u_textureSize", [actualWidth, actualHeight], "2f");
    GPU.setUniformForProgram("render", "u_obstacleRad", obstacleRad, "1f");

    // boundary
    GPU.setProgram("boundary");
    GPU.setUniformForProgram("boundary", "u_textureSize", [width, height], "2f");
    GPU.setUniformForProgram("boundary", "u_obstaclePosition", [obstaclePosition[0]*width/actualWidth, obstaclePosition[1]*height/actualHeight], "2f");
    GPU.setUniformForProgram("boundary", "u_obstacleRad", obstacleRad*width/actualWidth, "1f");

    var velocity = new Uint16Array(width*height*4);
    for (var i=0;i<height;i++){
        for (var j=0;j<width;j++){
            var index = 4*(i*width+j);
            velocity[index] = 10.0;
        }
    }

    // velocity
    GPU.initTextureFromData("velocity", width, height, "HALF_FLOAT", null, true);
    GPU.setProgram("force");
    GPU.initFrameBufferForTexture("velocity", true);

    // nextVelocity
    GPU.initTextureFromData("nextVelocity", width, height, "HALF_FLOAT", null, true);
    GPU.initFrameBufferForTexture("nextVelocity", true);

    // resetVelocity
    GPU.step("resetVelocity", [], "velocity");
    GPU.step("resetVelocity", [], "nextVelocity");

    // velocityDivergence
    GPU.initTextureFromData("velocityDivergence", width, height, "HALF_FLOAT", new Uint16Array(width*height*4), true);
    GPU.initFrameBufferForTexture("velocityDivergence", true);

    // pressure
    GPU.initTextureFromData("pressure", width, height, "HALF_FLOAT", new Uint16Array(width*height*4), true);
    GPU.initFrameBufferForTexture("pressure", true);

    // nextPressure
    GPU.initTextureFromData("nextPressure", width, height, "HALF_FLOAT", new Uint16Array(width*height*4), true);
    GPU.initFrameBufferForTexture("nextPressure", true);

    // var numCols = Math.floor(actualHeight/10);
    // if (numCols%2 == 1) numCols--;
    // var numPx = actualHeight/numCols;

    // var material = new Uint16Array(actualWidth*actualHeight*4);
    // for (var i=0;i<actualHeight;i++){
    //     for (var j=0;j<actualWidth;j++){
    //         var index = 4*(i*actualWidth+j);
    //         if (j==0 && Math.floor((i-2)/numPx)%2==0) material[index] = 1.0;
    //     }
    // }

    // material
    GPU.initTextureFromData("material", actualWidth, actualHeight, "HALF_FLOAT", null, true);
    GPU.initFrameBufferForTexture("material", true);

    // nextMaterial
    GPU.initTextureFromData("nextMaterial", actualWidth, actualHeight, "HALF_FLOAT", null, true);
    GPU.initFrameBufferForTexture("nextMaterial", true);

    paused = false;
}

function render(){
    // paused 가 아니면 (paused 인 경우는 화면 리사이즈만 해당됨.)
    if (!paused) {
        
        // 속도 설정
        GPU.setSize(width, height);
        GPU.step("advectVel", ["velocity", "velocity"], "nextVelocity");
        GPU.setProgram("boundary");
        if (movingObstacle) GPU.setUniformForProgram("boundary" ,"u_obstaclePosition", 
            [obstaclePosition[0]*width/actualWidth, 
            obstaclePosition[1]*height/actualHeight], 
        "2f");
        GPU.setUniformForProgram("boundary", "u_scale", -1, "1f");
        GPU.step("boundary", ["nextVelocity"], "velocity");

        // diffuse velocity
        GPU.setProgram("jacobi");
        var alpha = dx*dx/((mu/rho)*dt);
        GPU.setUniformForProgram("jacobi", "u_alpha", alpha, "1f");
        GPU.setUniformForProgram("jacobi", "u_reciprocalBeta", 1/(4+alpha), "1f");
        for (var i=0;i<10;i++){
            GPU.step("jacobi", ["velocity", "velocity"], "nextVelocity");
            GPU.step("jacobi", ["nextVelocity", "nextVelocity"], "velocity");
        }

        //apply force
        // if (mouseEnable){
        //     GPU.setProgram("force");
        //     GPU.setUniformForProgram("force", "u_mouseCoord", [mouseCoordinates[0]*width/actualWidth, mouseCoordinates[1]*height/actualHeight], "2f");
        //     GPU.setUniformForProgram("force", "u_mouseDir", [2*(mouseCoordinates[0]-lastMouseCoordinates[0])/scale,
        //         2*(mouseCoordinates[1]-lastMouseCoordinates[1])/scale], "2f");
        //     GPU.step("force", ["velocity"], "nextVelocity");
        //     GPU.setProgram("boundary");
        //     GPU.setUniformForProgram("boundary", "u_scale", -1, "1f");
        //     GPU.step("boundary", ["nextVelocity"], "velocity");
        // }

        // compute pressure
        GPU.step("diverge", ["velocity"], "velocityDivergence");//calc velocity divergence
        GPU.setProgram("jacobi");
        GPU.setUniformForProgram("jacobi", "u_alpha", -dx*dx, "1f");
        GPU.setUniformForProgram("jacobi", "u_reciprocalBeta", 1/4, "1f");
        for (var i=0;i<20;i++){
            GPU.step("jacobi", ["velocityDivergence", "pressure"], "nextPressure");//diffuse velocity
            GPU.step("jacobi", ["velocityDivergence", "nextPressure"], "pressure");//diffuse velocity
        }
        GPU.setProgram("boundary");
        GPU.setUniformForProgram("boundary", "u_scale", 1, "1f");
        GPU.step("boundary", ["pressure"], "nextPressure");
        GPU.swapTextures("nextPressure", "pressure");

        // subtract pressure gradient
        GPU.step("gradientSubtraction", ["velocity", "pressure"], "nextVelocity");
        GPU.setProgram("boundary");
        GPU.setUniformForProgram("boundary", "u_scale", -1, "1f");
        GPU.step("boundary", ["nextVelocity"], "velocity");

        // move material
        GPU.setSize(actualWidth, actualHeight);
        GPU.step("advectMat", ["velocity", "material"], "nextMaterial");
        
        if (movingObstacle) {
            GPU.setProgram("boundary");
            GPU.setUniformForProgram("boundary", "u_obstaclePosition", 
                [obstaclePosition[0]*width/actualWidth, 
                obstaclePosition[1]*height/actualHeight], 
            "2f");
            
            GPU.setProgram("render");
            GPU.setUniformForProgram("render", "u_obstaclePosition", 
                [obstaclePosition[0], 
                obstaclePosition[1]], 
            "2f");
        } 
        
        if (obstacleEnabled) {
            // 장애물이 활성화된 경우
            GPU.setProgram("boundary");
            GPU.setUniformForProgram("boundary", "u_obstaclePosition", 
                [obstaclePosition[0]*width/actualWidth, 
                obstaclePosition[1]*height/actualHeight], 
            "2f");
            GPU.setProgram("render");
            GPU.setUniformForProgram("render", "u_obstaclePosition", 
                [obstaclePosition[0], 
                obstaclePosition[1]], 
            "2f");
        } else {
            // 장애물이 비활성화된 경우
            GPU.setProgram("boundary");
            GPU.setUniformForProgram("boundary", "u_obstaclePosition", 
                [-9999, -9999], 
            "2f");
            GPU.setProgram("render");
            GPU.setUniformForProgram("render", "u_obstaclePosition", 
                [-9999, -9999], 
            "2f");
        }
        
        GPU.step("render", ["nextMaterial"]);
        GPU.swapTextures("nextMaterial", "material");

    } else resetWindow();

    window.requestAnimationFrame(render);
}

// 장애물 상태를 토글하는 함수
function toggleObstacle() {
    obstacleEnabled = !obstacleEnabled;
}

// 장애물 위치 업데이트 함수
function updateObstaclePosition(){
    if (movingObstacle) obstaclePosition = mouseCoordinates; // 장애물이 움직이고 있는경우, 장애물의 위치를 현재 마우스 좌표로 업데이트
}

// 마우스 이동 이벤트 핸들러
function onMouseMove(e){
    lastMouseCoordinates = mouseCoordinates; // 마지막 마우스 좌표값 <- 현재 마우스 좌표값
    const canvasRect = canvas.getBoundingClientRect();
    mouseCoordinates = [e.clientX-canvasRect.left, actualHeight-e.clientY+canvasRect.top]; // 현재 마우스 좌표값 <- 위치값 수정 (container 태그안에 넣었기 때문에 위치변경되었음)
    updateObstaclePosition(); // 장애물 위치 업데이트
}

// 터치 이동 이벤트 핸들러
function onTouchMove(e){
    e.preventDefault(); // 기본 터치 동작 방지 (스크롤 방지등)
    lastMouseCoordinates = mouseCoordinates; // 마지막 마우스 좌표값 <- 현재 마우스 좌표값
    mouseCoordinates = [touch.pageX, actualHeight-touch.pageY]; // 현재 터치 좌표로 업데이트
    updateObstaclePosition();
}

// 마우스 버튼 눌림 이벤트 핸들러
function onMouseDown(){

    // 마우스 좌표와 장애물 좌표 간의 거리 계산
    var distToObstacle = [mouseCoordinates[0]-obstaclePosition[0], mouseCoordinates[1]-obstaclePosition[1]];

    // 장애물 반지름 내에 마우스가 있는지 확인
    if (distToObstacle[0]*distToObstacle[0] + distToObstacle[1]*distToObstacle[1] < obstacleRad*obstacleRad){
        movingObstacle = true; // 장애물 움직임 O
        mouseEnable = false; // 마우스 움직임 O
    } else {
        mouseEnable = true; // 마우스 움직임 O
        movingObstacle = false; // 장애물 움직임 X
    }
}

// 마우스 버튼 뗌 이벤트 핸들러
function onMouseUp(){
    movingObstacle = false; // 장애물 움직임 X
    mouseEnable = false; // 마우스 움직임 X
}