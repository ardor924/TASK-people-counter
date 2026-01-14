from collections import defaultdict

class VotingProcessor:
    def __init__(self, max_history=30):
        # max_history를 20 -> 30으로 늘려 더 오랫동안 안정성을 유지하도록 변경
        self.max_history = max_history
        self.track_history = {}

    def update(self, track_id, g_val, g_conf, c_val, c_conf):
        """ 
        [Smart Update] 
        - Unknown이거나 신뢰도가 너무 낮으면(0.4 미만) 투표함에 넣지 않음 (노이즈 필터링)
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'gender': [], 
                'color': [],
                'last_valid_gender': ('Unknown', 0.0), # 관성 유지를 위한 마지막 유효값
                'last_valid_color': ('Unknown', 0.0)
            }
        
        # 성별 데이터 추가 (Unknown 제외, 신뢰도 필터)
        if g_val != "Unknown" and g_conf > 0.4:
            self.track_history[track_id]['gender'].append((g_val, g_conf))
            # 유효값 갱신
            self.track_history[track_id]['last_valid_gender'] = (g_val, g_conf)
            
        # 색상 데이터 추가 (Unknown 제외, 신뢰도 필터)
        if c_val != "Unknown" and c_conf > 0.4:
            self.track_history[track_id]['color'].append((c_val, c_conf))
            # 유효값 갱신
            self.track_history[track_id]['last_valid_color'] = (c_val, c_conf)

        # 히스토리 길이 제한 (Sliding Window)
        if len(self.track_history[track_id]['gender']) > self.max_history:
            self.track_history[track_id]['gender'].pop(0)
        
        if len(self.track_history[track_id]['color']) > self.max_history:
            self.track_history[track_id]['color'].pop(0)

    def get_result(self, track_id):
        """ 
        [Weighted Voting] 
        단순 빈도수가 아니라, 신뢰도의 합(Sum of Confidence)이 가장 높은 속성을 선택 
        """
        if track_id not in self.track_history:
            return {
                'gender': 'Unknown', 'gender_conf': 0.0,
                'color': 'Unknown', 'color_conf': 0.0
            }

        target = self.track_history[track_id]
        
        # 1. 성별 결정
        g_res, g_conf = self._weighted_vote(target['gender'])
        if g_res == "Unknown": # 히스토리가 비었으면 마지막 유효값 사용 (관성)
            g_res, g_conf = target['last_valid_gender']

        # 2. 색상 결정
        c_res, c_conf = self._weighted_vote(target['color'])
        if c_res == "Unknown": 
            c_res, c_conf = target['last_valid_color']

        return {
            'gender': g_res, 'gender_conf': g_conf,
            'color': c_res, 'color_conf': c_conf
        }

    def _weighted_vote(self, history_list):
        """ 가중 투표 로직 """
        if not history_list:
            return "Unknown", 0.0
        
        score_board = defaultdict(float)
        count_board = defaultdict(int)
        
        for val, conf in history_list:
            # 신뢰도를 점수로 누적 (가중치)
            score_board[val] += conf
            count_board[val] += 1
            
        if not score_board:
            return "Unknown", 0.0
            
        # 점수(신뢰도 합)가 가장 높은 속성 선택
        best_val = max(score_board, key=score_board.get)
        
        # 평균 신뢰도 계산
        avg_conf = score_board[best_val] / count_board[best_val]
        
        return best_val, avg_conf

    def clear(self):
        self.track_history.clear()