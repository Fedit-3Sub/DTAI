document.addEventListener('DOMContentLoaded', function() {
    const settingsModal = document.getElementById('settingsModal');
    const dtSlider = document.getElementById('dtSlider');
    const dtValue = document.getElementById('dtValue');
    const densitySlider = document.getElementById('densitySlider');
    const densityValue = document.getElementById('densityValue');
    const viscositySlider = document.getElementById('viscositySlider');
    const viscosityValue = document.getElementById('viscosityValue');
    const obstacleRadiusSlider = document.getElementById('obstacleRadiusSlider');
    const obstacleRadiusValue = document.getElementById('obstacleRadiusValue');


    // 버튼 관련 설정 이벤트 ------------------------------------------------------------------------
    // 옵션 버튼 이벤트
    document.getElementById('openSettings').addEventListener('click', function () {
        settingsModal.style.display = 'block';
    
        // 화면 중앙에 모달 위치 설정 (초기화 수행) -> 버튼 누를때 뜨는 위치 설정
        const modalWidth = settingsModal.offsetWidth;
        const modalHeight = settingsModal.offsetHeight;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const centerX = (viewportWidth - modalWidth) / 2;
        const centerY = (viewportHeight - modalHeight) / 2;
        settingsModal.style.left = `${centerX}px`;
        settingsModal.style.top = `${centerY}px`;
    });

    // 닫기 버튼 이벤트
    document.getElementById('closeSettings').addEventListener('click', function () {
        settingsModal.style.display = 'none';
    });

    // default 버튼 이벤트
    document.getElementById('defaultSettings').addEventListener('click', function () {
        dtSlider.value = 1;
        densitySlider.value = 1;
        viscositySlider.value = 1;
        obstacleRadiusSlider.value = 30;
        window.updateProperties({
            dt: 1,
            rho: 1,
            mu: 1,
            obstacleRad: 30
        });
    });


    // 슬라이더 값 변경 이벤트 ------------------------------------------------------------------------
    // 동기화 (입력창 변경시 슬라이더 동기화)
    function syncSlider(slider, input) {
        slider.value = input.value;
        slider.dispatchEvent(new Event('input')); // 슬라이더 이벤트 강제 호출
    }

    // 동기화 (슬라이더 변경시 입력창 동기화)
    function syncInput(input, slider) {
        input.value = slider.value;
    }

    // Time Step 슬라이더 이벤트
    dtSlider.addEventListener('input', function () {
        syncInput(dtValue, dtSlider);
        window.updateProperties({dt:dtSlider.value});
    });

    dtValue.addEventListener('change', function () {
        syncSlider(dtSlider, dtValue);
    });

    // Density 슬라이더 이벤트
    densitySlider.addEventListener('input', function () {
        syncInput(densityValue, densitySlider);
        window.updateProperties({rho:densitySlider.value});
    });

    densityValue.addEventListener('change', function () {
        syncSlider(densitySlider, densityValue);
    });

    // Viscosity 슬라이더 이벤트
    viscositySlider.addEventListener('input', function () {
        syncInput(viscosityValue, viscositySlider);
        window.updateProperties({mu:viscositySlider.value});
    });

    viscosityValue.addEventListener('change', function () {
        syncSlider(viscositySlider, viscosityValue);
    });

    // Obstacle Radius 슬라이더 이벤트
    obstacleRadiusSlider.addEventListener('input', function () {
        syncInput(obstacleRadiusValue, obstacleRadiusSlider);
        window.updateProperties({obstacleRad:obstacleRadiusSlider.value});
    });

    obstacleRadiusValue.addEventListener('change', function () {
        syncSlider(obstacleRadiusSlider, obstacleRadiusValue);
    });



    // 옵션 모달 드래그 관련 이벤트 설정 ------------------------------------------------------------
    // 드래그 시작
    settingsModal.addEventListener("mousedown", (e) => {
        
        // 드래그 True 설정
        isDragging = true;

        // 드래그 중 커서 스타일 변경
        settingsModal.style.cursor = "grabbing";

        // 초기 위치값 (마우스 클릭한 위치)
        startX = e.clientX;
        startY = e.clientY;
    
        // 모달 초기위치 저장 (좌측 상단)
        modalX = parseInt(settingsModal.style.left);
        modalY = parseInt(settingsModal.style.top);
    });
    
    // 마우스 이동 (드래그 중)
    document.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
    
        const deltaX = e.clientX - startX;
        const deltaY = e.clientY - startY;
    
        // 모달 위치 업데이트
        settingsModal.style.left = `${modalX + deltaX}px`;
        settingsModal.style.top = `${modalY + deltaY}px`;
    });
    
    // 마우스 버튼 떼기 (드래그 종료)
    document.addEventListener("mouseup", () => {
        isDragging = false;
        settingsModal.style.cursor = "grab"; // 기본 커서로 복구
    });

});