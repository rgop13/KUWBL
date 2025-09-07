import gradio as gr
import json
import re

def load_jsonl_data(file_path):
    """JSONL 파일을 로드하고 데이터를 반환합니다."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"줄 {line_num}에서 JSON 파싱 오류: {e}")
                    continue
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
    
    return data

def extract_thinking_process(assistant_content):
    """assistant 응답에서 thinking 과정과 실제 출력을 분리합니다."""
    # <think>와 </think> 사이의 내용을 찾기
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_match = re.search(thinking_pattern, assistant_content, re.DOTALL)
    
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        # thinking 부분을 제거한 나머지가 실제 출력
        actual_output = re.sub(thinking_pattern, '', assistant_content, flags=re.DOTALL).strip()
    else:
        thinking_content = "thinking 과정이 없습니다."
        actual_output = assistant_content
    
    return thinking_content, actual_output

def get_conversation_data(data, index):
    """지정된 인덱스의 대화 데이터를 추출합니다."""
    if not data or index < 0 or index >= len(data):
        return "데이터 없음", "데이터 없음", "데이터 없음", "데이터 없음", 0, 0, 0, 0
    
    conversation = data[index]
    
    # 기본값 설정
    system_prompt = "시스템 프롬프트가 없습니다."
    user_input = "사용자 입력이 없습니다."
    thinking_process = "thinking 과정이 없습니다."
    assistant_output = "assistant 출력이 없습니다."
    
    # query_and_response에서 데이터 추출
    if 'query_and_response' in conversation:
        for message in conversation['query_and_response']:
            # 'from' 또는 'role' 키를 모두 지원
            message_from = message.get('from') or message.get('role')
            if message_from == 'system':
                system_prompt = message.get('content', system_prompt)
            elif message_from == 'user':
                user_input = message.get('content', user_input)
            elif message_from == 'assistant':
                assistant_content = message.get('content', '')
                thinking_process, assistant_output = extract_thinking_process(assistant_content)
    
    # 문자 수 계산
    system_char_count = len(system_prompt)
    user_char_count = len(user_input)
    thinking_char_count = len(thinking_process)
    assistant_char_count = len(assistant_output)
    
    return system_prompt, user_input, thinking_process, assistant_output, system_char_count, user_char_count, thinking_char_count, assistant_char_count

def get_comparison_data(original_data, current_data, index):
    """두 데이터셋의 지정된 인덱스 데이터를 비교용으로 추출합니다."""
    # 원본 데이터
    orig_system, orig_user, orig_thinking, orig_output, orig_sys_count, orig_user_count, orig_think_count, orig_assist_count = get_conversation_data(original_data, index)
    
    # 현재 데이터
    curr_system, curr_user, curr_thinking, curr_output, curr_sys_count, curr_user_count, curr_think_count, curr_assist_count = get_conversation_data(current_data, index)
    
    return (orig_system, orig_user, orig_thinking, orig_output, orig_sys_count, orig_user_count, orig_think_count, orig_assist_count,
            curr_system, curr_user, curr_thinking, curr_output, curr_sys_count, curr_user_count, curr_think_count, curr_assist_count)

def update_conversation(data, index):
    """대화 인덱스가 변경될 때 호출되는 함수입니다."""
    system_prompt, user_input, thinking_process, assistant_output, system_char_count, user_char_count, thinking_char_count, assistant_char_count = get_conversation_data(data, index)
    
    # 인덱스 정보 추가
    total_count = len(data) if data else 0
    index_info = f"대화 {index + 1} / {total_count}"
    
    # 문자 수 정보 텍스트 생성
    system_char_info = f"시스템 프롬프트 ({system_char_count:,}자)"
    user_char_info = f"사용자 입력 ({user_char_count:,}자)"
    thinking_char_info = f"Thinking 과정 ({thinking_char_count:,}자)"
    assistant_char_info = f"Assistant 출력 ({assistant_char_count:,}자)"
    
    return system_prompt, user_input, thinking_process, assistant_output, index_info, system_char_info, user_char_info, thinking_char_info, assistant_char_info

def update_comparison(original_data, current_data, index):
    """비교 인덱스가 변경될 때 호출되는 함수입니다."""
    (orig_system, orig_user, orig_thinking, orig_output, orig_sys_count, orig_user_count, orig_think_count, orig_assist_count,
     curr_system, curr_user, curr_thinking, curr_output, curr_sys_count, curr_user_count, curr_think_count, curr_assist_count) = get_comparison_data(original_data, current_data, index)
    
    # 인덱스 정보 추가
    orig_total = len(original_data) if original_data else 0
    curr_total = len(current_data) if current_data else 0
    index_info = f"대화 {index + 1} / {min(orig_total, curr_total)}"
    
    # 문자 수 정보 텍스트 생성
    orig_sys_char_info = f"원본 시스템 ({orig_sys_count:,}자)"
    orig_user_char_info = f"원본 사용자 ({orig_user_count:,}자)"
    orig_think_char_info = f"원본 Think ({orig_think_count:,}자)"
    orig_assist_char_info = f"원본 Response ({orig_assist_count:,}자)"
    
    curr_sys_char_info = f"현재 시스템 ({curr_sys_count:,}자)"
    curr_user_char_info = f"현재 사용자 ({curr_user_count:,}자)"
    curr_think_char_info = f"현재 Think ({curr_think_count:,}자)"
    curr_assist_char_info = f"현재 Response ({curr_assist_count:,}자)"
    
    return (orig_system, orig_user, orig_thinking, orig_output, 
            curr_system, curr_user, curr_thinking, curr_output,
            index_info, orig_sys_char_info, orig_user_char_info, orig_think_char_info, orig_assist_char_info,
            curr_sys_char_info, curr_user_char_info, curr_think_char_info, curr_assist_char_info)

def create_gradio_interface(original_jsonl_path, current_jsonl_path):
    """비교용 Gradio 인터페이스를 생성합니다."""
    
    # 데이터 로드
    print("원본 데이터를 로드하는 중...")
    original_data = load_jsonl_data(original_jsonl_path)
    print(f"원본: 총 {len(original_data)}개의 대화를 로드했습니다.")
    
    print("현재 데이터를 로드하는 중...")
    current_data = load_jsonl_data(current_jsonl_path)
    print(f"현재: 총 {len(current_data)}개의 대화를 로드했습니다.")
    
    total_conversations = min(len(original_data), len(current_data))
    print(f"비교 가능한 대화 수: {total_conversations}개")
    
    # 커스텀 CSS 스타일
    custom_css = """
    /* 원본 데이터 스타일 */
    #orig_system_prompt textarea, #orig_user_input textarea, #orig_thinking textarea, #orig_output textarea {
        border-left: 4px solid #3498db !important;
    }
    
    /* 현재 데이터 스타일 */
    #curr_system_prompt textarea, #curr_user_input textarea, #curr_thinking textarea, #curr_output textarea {
        border-left: 4px solid #e74c3c !important;
    }
    
    /* 시스템 프롬프트 스타일 */
    #orig_system_prompt textarea, #curr_system_prompt textarea {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 10px !important;
        line-height: 1.5 !important;
    }
    
    /* 사용자 입력 스타일 */
    #orig_user_input textarea, #curr_user_input textarea {
        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
        font-size: 10px !important;
        line-height: 1.6 !important;
    }
    
    /* Thinking 과정 스타일 */
    #orig_thinking textarea, #curr_thinking textarea {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 10px !important;
        line-height: 1.4 !important;
        color: #555 !important;
    }
    
    /* Assistant 출력 스타일 */
    #orig_output textarea, #curr_output textarea {
        font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
        font-size: 10px !important;
        line-height: 1.6 !important;
        color: #2c3e50 !important;
    }
    
    /* 글꼴 크기 조정 슬라이더 컨테이너 */
    .font-controls {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* 비교 헤더 스타일 */
    .comparison-header {
        text-align: center;
        background: linear-gradient(90deg, #3498db 50%, #e74c3c 50%);
        color: white;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    """
    
    # Gradio 인터페이스 구성
    with gr.Blocks(title="데이터 비교 도구", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 📊 JSONL 데이터 비교 도구")
        gr.Markdown(f"**원본 파일**: `{original_jsonl_path}`")
        gr.Markdown(f"**현재 파일**: `{current_jsonl_path}`")
        gr.Markdown(f"**비교 가능한 대화 수**: {total_conversations}")
        
        # 색상 범례
        with gr.Row():
            gr.Markdown("🔵 **원본 데이터** | 🔴 **현재 데이터**")
        
        # 글꼴 크기 조정 섹션
        with gr.Accordion("🎨 글꼴 설정", open=False):
            with gr.Row(elem_classes="font-controls"):
                system_font_size = gr.Slider(
                    minimum=10, maximum=24, step=1, value=14,
                    label="시스템 프롬프트 글꼴 크기", interactive=True
                )
                user_font_size = gr.Slider(
                    minimum=10, maximum=24, step=1, value=15,
                    label="사용자 입력 글꼴 크기", interactive=True
                )
            with gr.Row(elem_classes="font-controls"):
                thinking_font_size = gr.Slider(
                    minimum=10, maximum=24, step=1, value=13,
                    label="Thinking 과정 글꼴 크기", interactive=True
                )
                assistant_font_size = gr.Slider(
                    minimum=10, maximum=24, step=1, value=15,
                    label="Assistant 출력 글꼴 크기", interactive=True
                )
        
        # 인덱스 선택 슬라이더
        with gr.Row():
            index_slider = gr.Slider(
                minimum=0, 
                maximum=max(0, total_conversations - 1), 
                step=1, 
                value=0, 
                label="대화 선택",
                interactive=True
            )
            index_info = gr.Textbox(
                value=f"대화 1 / {total_conversations}",
                label="현재 위치",
                interactive=False,
                max_lines=1
            )
        
        # 네비게이션 버튼
        with gr.Row():
            prev_btn = gr.Button("⬅️ 이전", variant="secondary", size="sm")
            next_btn = gr.Button("➡️ 다음", variant="secondary", size="sm")
        
        # 시스템 프롬프트 비교 섹션
        with gr.Row():
            orig_sys_char_label = gr.Markdown("## 🔵 원본 시스템 프롬프트 (0자)")
            curr_sys_char_label = gr.Markdown("## 🔴 현재 시스템 프롬프트 (0자)")
        with gr.Row():
            orig_system_box = gr.Textbox(
                label="원본 시스템 프롬프트",
                lines=5,
                max_lines=10,
                interactive=False,
                show_copy_button=True,
                elem_id="orig_system_prompt"
            )
            curr_system_box = gr.Textbox(
                label="현재 시스템 프롬프트",
                lines=5,
                max_lines=10,
                interactive=False,
                show_copy_button=True,
                elem_id="curr_system_prompt"
            )
        
        # 사용자 입력 비교 섹션
        with gr.Row():
            orig_user_char_label = gr.Markdown("## 🔵 원본 사용자 입력 (0자)")
            curr_user_char_label = gr.Markdown("## 🔴 현재 사용자 입력 (0자)")
        with gr.Row():
            orig_user_box = gr.Textbox(
                label="원본 사용자 질문/요청",
                lines=5,
                max_lines=10,
                interactive=False,
                show_copy_button=True,
                elem_id="orig_user_input"
            )
            curr_user_box = gr.Textbox(
                label="현재 사용자 질문/요청",
                lines=5,
                max_lines=10,
                interactive=False,
                show_copy_button=True,
                elem_id="curr_user_input"
            )
        
        # Thinking 과정 비교 섹션
        with gr.Row():
            orig_think_char_label = gr.Markdown("## 🔵 원본 Think 과정 (0자)")
            curr_think_char_label = gr.Markdown("## 🔴 현재 Think 과정 (0자)")
        with gr.Row():
            orig_thinking_box = gr.Textbox(
                label="원본 내부 사고 과정",
                lines=15,
                max_lines=25,
                interactive=False,
                show_copy_button=True,
                placeholder="thinking 과정이 길 수 있습니다. 스크롤하여 전체 내용을 확인하세요.",
                elem_id="orig_thinking"
            )
            curr_thinking_box = gr.Textbox(
                label="현재 내부 사고 과정",
                lines=15,
                max_lines=25,
                interactive=False,
                show_copy_button=True,
                placeholder="thinking 과정이 길 수 있습니다. 스크롤하여 전체 내용을 확인하세요.",
                elem_id="curr_thinking"
            )
        
        # Response 출력 비교 섹션
        with gr.Row():
            orig_output_char_label = gr.Markdown("## 🔵 원본 Response 출력 (0자)")
            curr_output_char_label = gr.Markdown("## 🔴 현재 Response 출력 (0자)")
        with gr.Row():
            orig_output_box = gr.Textbox(
                label="원본 Response (thinking 제외)",
                lines=15,
                max_lines=25,
                interactive=False,
                show_copy_button=True,
                elem_id="orig_output"
            )
            curr_output_box = gr.Textbox(
                label="현재 Response (thinking 제외)",
                lines=15,
                max_lines=25,
                interactive=False,
                show_copy_button=True,
                elem_id="curr_output"
            )
        
        # 초기 데이터 로드
        if original_data and current_data:
            (orig_system, orig_user, orig_thinking, orig_output, 
             curr_system, curr_user, curr_thinking, curr_output,
             index_info_text, orig_sys_char_info, orig_user_char_info, orig_think_char_info, orig_assist_char_info,
             curr_sys_char_info, curr_user_char_info, curr_think_char_info, curr_assist_char_info) = update_comparison(original_data, current_data, 0)
            
            # 초기 값 설정
            orig_system_box.value = orig_system
            orig_user_box.value = orig_user
            orig_thinking_box.value = orig_thinking
            orig_output_box.value = orig_output
            
            curr_system_box.value = curr_system
            curr_user_box.value = curr_user
            curr_thinking_box.value = curr_thinking
            curr_output_box.value = curr_output
            
            # 초기 문자 수 라벨 업데이트
            orig_sys_char_label.value = f"## 🔵 {orig_sys_char_info}"
            orig_user_char_label.value = f"## 🔵 {orig_user_char_info}"
            orig_think_char_label.value = f"## 🔵 {orig_think_char_info}"
            orig_output_char_label.value = f"## 🔵 {orig_assist_char_info}"
            
            curr_sys_char_label.value = f"## 🔴 {curr_sys_char_info}"
            curr_user_char_label.value = f"## 🔴 {curr_user_char_info}"
            curr_think_char_label.value = f"## 🔴 {curr_think_char_info}"
            curr_output_char_label.value = f"## 🔴 {curr_assist_char_info}"
        
        # 이벤트 핸들러 설정
        def on_index_change(index):
            return update_comparison(original_data, current_data, int(index))
        
        index_slider.change(
            fn=on_index_change,
            inputs=[index_slider],
            outputs=[orig_system_box, orig_user_box, orig_thinking_box, orig_output_box, 
                     curr_system_box, curr_user_box, curr_thinking_box, curr_output_box,
                     index_info, orig_sys_char_label, orig_user_char_label, orig_think_char_label, orig_output_char_label,
                     curr_sys_char_label, curr_user_char_label, curr_think_char_label, curr_output_char_label]
        )
        
        def go_previous(current_index):
            new_index = max(0, current_index - 1)
            results = update_comparison(original_data, current_data, new_index)
            return [new_index] + list(results)
            
        def go_next(current_index):
            new_index = min(total_conversations - 1, current_index + 1)
            results = update_comparison(original_data, current_data, new_index)
            return [new_index] + list(results)
        
        prev_btn.click(
            fn=go_previous,
            inputs=[index_slider],
            outputs=[index_slider, orig_system_box, orig_user_box, orig_thinking_box, orig_output_box, 
                     curr_system_box, curr_user_box, curr_thinking_box, curr_output_box,
                     index_info, orig_sys_char_label, orig_user_char_label, orig_think_char_label, orig_output_char_label,
                     curr_sys_char_label, curr_user_char_label, curr_think_char_label, curr_output_char_label]
        )
        
        next_btn.click(
            fn=go_next,
            inputs=[index_slider],
            outputs=[index_slider, orig_system_box, orig_user_box, orig_thinking_box, orig_output_box, 
                     curr_system_box, curr_user_box, curr_thinking_box, curr_output_box,
                     index_info, orig_sys_char_label, orig_user_char_label, orig_think_char_label, orig_output_char_label,
                     curr_sys_char_label, curr_user_char_label, curr_think_char_label, curr_output_char_label]
        )
        
        # 글꼴 크기 변경 JavaScript 함수
        font_change_js = """
        function(system_size, user_size, thinking_size, assistant_size) {
            setTimeout(function() {
                // 원본 데이터 요소들
                const origSystemTextarea = document.querySelector('#orig_system_prompt textarea');
                const origUserTextarea = document.querySelector('#orig_user_input textarea');
                const origThinkingTextarea = document.querySelector('#orig_thinking textarea');
                const origOutputTextarea = document.querySelector('#orig_output textarea');
                
                // 현재 데이터 요소들
                const currSystemTextarea = document.querySelector('#curr_system_prompt textarea');
                const currUserTextarea = document.querySelector('#curr_user_input textarea');
                const currThinkingTextarea = document.querySelector('#curr_thinking textarea');
                const currOutputTextarea = document.querySelector('#curr_output textarea');
                
                // 원본 데이터 글꼴 크기 설정
                if (origSystemTextarea) origSystemTextarea.style.fontSize = system_size + 'px';
                if (origUserTextarea) origUserTextarea.style.fontSize = user_size + 'px';
                if (origThinkingTextarea) origThinkingTextarea.style.fontSize = thinking_size + 'px';
                if (origOutputTextarea) origOutputTextarea.style.fontSize = assistant_size + 'px';
                
                // 현재 데이터 글꼴 크기 설정
                if (currSystemTextarea) currSystemTextarea.style.fontSize = system_size + 'px';
                if (currUserTextarea) currUserTextarea.style.fontSize = user_size + 'px';
                if (currThinkingTextarea) currThinkingTextarea.style.fontSize = thinking_size + 'px';
                if (currOutputTextarea) currOutputTextarea.style.fontSize = assistant_size + 'px';
            }, 100);
            return [system_size, user_size, thinking_size, assistant_size];
        }
        """
        
        # 글꼴 크기 변경 이벤트 핸들러 (각 슬라이더별로 개별 설정)
        system_font_size.change(
            fn=None,
            inputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            outputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            js=font_change_js
        )
        
        user_font_size.change(
            fn=None,
            inputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            outputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            js=font_change_js
        )
        
        thinking_font_size.change(
            fn=None,
            inputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            outputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            js=font_change_js
        )
        
        assistant_font_size.change(
            fn=None,
            inputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            outputs=[system_font_size, user_font_size, thinking_font_size, assistant_font_size],
            js=font_change_js
        )
        
        # 사용법 안내
        with gr.Accordion("📖 사용법", open=False):
            gr.Markdown("""
            ### 사용 방법
            1. **슬라이더 조작**: 상단의 슬라이더를 움직여 원하는 대화를 선택하세요.
            2. **버튼 네비게이션**: '이전'/'다음' 버튼으로 순차적으로 대화를 탐색하세요.
            3. **복사 기능**: 각 텍스트 박스 우측의 복사 버튼을 클릭하여 내용을 클립보드에 복사할 수 있습니다.
            4. **스크롤**: Think 과정과 Response 출력이 길 수 있으니 스크롤하여 전체 내용을 확인하세요.
            5. **글꼴 설정**: '🎨 글꼴 설정' 아코디언을 열어 각 섹션별로 글꼴 크기를 조정할 수 있습니다.
            6. **비교 분석**: 좌측(🔵)은 원본 데이터, 우측(🔴)은 현재 데이터로 동시에 비교할 수 있습니다.
            
            ### 데이터 구조
            - **시스템 프롬프트**: AI의 행동을 지시하는 초기 설정 (고정폭 폰트)
            - **사용자 입력**: 사용자가 제출한 질문이나 요청 (한글 최적화 폰트)
            - **Think 과정**: AI의 내부 사고 과정 (`<think>` 태그 내부, 고정폭 폰트)
            - **Response 출력**: 사용자에게 제공되는 최종 답변 (한글 최적화 폰트)
            
            ### 비교 기능
            - **🔵 원본 데이터**: 좌측에 파란색 경계선으로 표시
            - **🔴 현재 데이터**: 우측에 빨간색 경계선으로 표시
            - **동시 비교**: 같은 인덱스의 원본과 현재 데이터를 나란히 비교
            
            ### 글꼴 특징
            - **시스템 프롬프트 & Think 과정**: 고정폭 폰트 (Consolas, Monaco, Courier New)로 코드나 구조화된 텍스트에 적합
            - **사용자 입력 & Response 출력**: 한글 최적화 폰트 (맑은 고딕, Apple SD Gothic Neo)로 가독성 향상
            - **글꼴 크기**: 10px~24px 범위에서 1px 단위로 조정 가능
            """)
    
    return demo

if __name__ == "__main__":
    original_jsonl_path = "/data_x/WBL/data/sampled_data/generated/safety_500_response_generated.jsonl"
    current_jsonl_path = "/data_x/WBL/data/sampled_data/generated/safety_500_response_generated_3.jsonl"

    demo = create_gradio_interface(original_jsonl_path, current_jsonl_path)
    demo.launch(
        share=False,            # 공개 링크 생성 안함
    )
