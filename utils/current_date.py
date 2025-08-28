from datetime import datetime

week_map = ["日", "一", "二", "三", "四", "五", "六"]


def get_current_date_info() -> str:
    """
    今天是 2025年8月15日 星期五
    """

    return f"今天是 {datetime.now().strftime('%Y-%m-%d')} 星期{week_map[int(datetime.now().strftime('%w'))]}"