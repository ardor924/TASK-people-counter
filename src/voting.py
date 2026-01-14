from collections import defaultdict, deque

class VotingProcessor:
    def __init__(self, max_history=30):
        self.max_history = max_history
        self.history = {}  # {track_id: {'gender': deque(), 'color': deque()}}

    def update(self, track_id, gender, gender_conf, color, color_conf):
        if track_id not in self.history:
            self.history[track_id] = {
                'gender': deque(maxlen=self.max_history),
                'color': deque(maxlen=self.max_history)
            }
        
        # [핵심] 'Unk'나 신뢰도가 너무 낮은 값은 투표함에 넣지 않음 (노이즈 필터링)
        if gender != "Unk" and gender_conf > 0.5:
            self.history[track_id]['gender'].append((gender, gender_conf))
            
        if color != "Unk" and color_conf > 0.5:
            self.history[track_id]['color'].append((color, color_conf))

    def get_result(self, track_id):
        if track_id not in self.history:
            return {'gender': "Unknown", 'gender_conf': 0.0, 
                    'color': "Unknown", 'color_conf': 0.0}

        # 성별 투표
        final_gender, final_g_conf = self._vote(self.history[track_id]['gender'])
        # 색상 투표
        final_color, final_c_conf = self._vote(self.history[track_id]['color'])

        return {
            'gender': final_gender,
            'gender_conf': final_g_conf,
            'color': final_color,
            'color_conf': final_c_conf
        }

    def _vote(self, queue):
        if not queue:
            return "Unknown", 0.0
        
        # 가중 투표: 빈도수 * 신뢰도 합산
        scores = defaultdict(float)
        counts = defaultdict(int)
        
        for val, conf in queue:
            scores[val] += conf
            counts[val] += 1
            
        if not scores:
            return "Unknown", 0.0
            
        # 점수가 가장 높은 속성 선택
        best_val = max(scores, key=scores.get)
        
        # 평균 신뢰도 계산
        avg_conf = scores[best_val] / counts[best_val]
        
        return best_val, avg_conf

    def clear(self):
        self.history.clear()