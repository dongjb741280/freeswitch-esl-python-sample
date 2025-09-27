
import time
try:
    import ESL  # 官方模块
except Exception:
    from freeswitchESL import ESL  # 回退

def on_esl_event(event):
    event_name = event.getHeader("Event-Name")
    if event_name.startswith("CHANNEL_"):
        callee_number = event.getHeader("Caller-Callee-ID-Number")
        caller_number = event.getHeader("Caller-Caller-ID-Number")
        if event_name == "CHANNEL_CREATE":
            print(f"Call initiated, caller: {caller_number}, callee: {callee_number}")
        elif event_name == "CHANNEL_BRIDGE":
            print(f"User transfer, caller: {caller_number}, callee: {callee_number}")
        elif event_name == "CHANNEL_ANSWER":
            print(f"User answered, caller: {caller_number}, callee: {callee_number}")
        elif event_name == "CHANNEL_HANGUP":
            response = event.getHeader("variable_current_application_response")
            hangup_cause = event.getHeader("Hangup-Cause")
            print(f"User hung up, caller: {caller_number}, callee: {callee_number}, response: {response}, hangup cause: {hangup_cause}")

def main():
    con = ESL.ESLconnection("192.168.1.4", "8022", "ClueCon")
    if not con.connected():
        print("Connection failed!")
        return

    con.events("plain", "all")

    while con.connected():
        event = con.recvEvent()
        if event:
            on_esl_event(event)

if __name__ == "__main__":
    main()
