Scenario: Give large api reference document as context and ask specific questions from beginning, middle and end. 
expectation: hallucination/wrong answers expected especially when retreiving info from middle of the context window. 

run1: 

Document length: 48530 characters
--------------------------------------------------
Enter your question about the API: does wifi successful connection event return the bssid of the access point?

Sending to meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
Yes, according to the documentation, a "Wi-Fi Successful Connection" event does return the BSSID of the access point. 

The structure for "Wi-Fi Successful Connection" contains:
- BSSID (string) → The MAC address of the access point.

This is confirmed in the generated JSON message for "Wi-Fi Successful Connection":
```json
{
    ...
    "headerValue": {
        ...
        "BSSID": "12:34:56:78:9A:BC",
        ...
    }
}
```

==================================================
Prompt tokens: 13892
Completion tokens: 107
Total tokens: 13999

observation: model was able to retreive info from the middle correctly.
model context window: 10M 
context tokens: 13892 


run2:
Document length: 48530 characters
--------------------------------------------------
Enter your question about the API: how many different types of events are there? 

Sending to meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
There are 37 different types of events.

Here is a list of the event types:

1. Wi-Fi Scan Complete
2. IP Disconnection
3. IP Failed Connection
4. IPv4 Address Established
5. IPv6 Address Established
6. OneX Authentication
7. Suboptimal Wi-Fi Driver Property
8. Wi-Fi Bad Coverage
9. Wi-Fi Channel Switch
10. Wi-Fi Co-channel Interference
11. Wi-Fi Connection Short Period Statistics
12. Connections Topology Information
13. Wi-Fi Missed Beacons
14. Wi-Fi Driver Properties Indication
15. Wi-Fi No Coverage
16. Wi-Fi Successful Connection
17. Wi-Fi Failed Connection
18. Wi-Fi Toggle State Indication
19. Wi-Fi Disconnected Antenna
20. Wi-Fi Device Short Period Statistics
21. Wi-Fi Roaming Report
22. Wi-Fi Connection Statistics
23. Wi-Fi Disconnection
24. Wi-Fi Technology Gap
25. Wi-Fi Driver Availability Indication
26. WLAN Service State Change Indication
27. Wi-Fi Security Indication
28. Wi-Fi Device Statistics

==================================================
Prompt tokens: 13887
Completion tokens: 231
Total tokens: 14118

observation: model said 37 different events but only listed 28? doc has 28 events as well so where did it get 37 from? 
next step: test with summarize then answer

run3: 
Document length: 48530 characters
--------------------------------------------------
Enter your question about the API: I asked you about number of different events to which you said the following RESPONSE:There are 37 different types of events.Here is a list of the event types:1. Wi-Fi Scan Complete2. IP Disconnection3. IP Failed Connection4. IPv4 Address Established5. IPv6 Address Established6. OneX Authentication7. Suboptimal Wi-Fi Driver Property8. Wi-Fi Bad Coverage9. Wi-Fi Channel Switch10. Wi-Fi Co-channel Interference11. Wi-Fi Connection Short Period Statistics12. Connections Topology Information13. Wi-Fi Missed Beacons14. Wi-Fi Driver Properties Indication15. Wi-Fi No Coverage16. Wi-Fi Successful Connection17. Wi-Fi Failed Connection18. Wi-Fi Toggle State Indication19. Wi-Fi Disconnected Antenna20. Wi-Fi Device Short Period Statistics21. Wi-Fi Roaming Report22. Wi-Fi Connection Statistics23. Wi-Fi Disconnection24. Wi-Fi Technology Gap25. Wi-Fi Driver Availability Indication26. WLAN Service State Change Indication27. Wi-Fi Security Indication28. Wi-Fi Device Statistics where did you
 get the number 37 from when the doc lists 28 and you also listed 28?

Sending to meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
You are correct that the documentation lists 28 different event types, not 37. I made an error in my previous response. The correct number of event types is 28, as listed:

1. Wi-Fi Scan Complete
2. IP Disconnection
3. IP Failed Connection
4. IPv4 Address Established
5. IPv6 Address Established
6. OneX Authentication
7. Suboptimal Wi-Fi Driver Property
8. Wi-Fi Bad Coverage
9. Wi-Fi Channel Switch
10. Wi-Fi Co-channel Interference
11. Wi-Fi Connection Short Period Statistics
12. Connections Topology Information
13. Wi-Fi Missed Beacons
14. Wi-Fi Driver Properties Indication
15. Wi-Fi No Coverage
16. Wi-Fi Successful Connection
17. Wi-Fi Failed Connection
18. Wi-Fi Toggle State Indication
19. Wi-Fi Disconnected Antenna
20. Wi-Fi Device Short Period Statistics
21. Wi-Fi Roaming Report
22. Wi-Fi Connection Statistics
23. Wi-Fi Disconnection
24. Wi-Fi Technology Gap
25. Wi-Fi Driver Availability Indication
26. WLAN Service State Change Indication
27. Wi-Fi Security Indication
28. Wi-Fi Device Statistics

I strive to provide accurate information, and I appreciate you bringing this to my attention. The correct number of different events is indeed 28.

==================================================
Prompt tokens: 14118
Completion tokens: 280
Total tokens: 14398

observation: model realized its mistake? or did it just state 28 because I prompted it? 

run4: try to trick the model
Loaded document from d:\Work\gen-ai\tasks\week-1\day-2\task1\prompt.txt
Document length: 48530 characters
--------------------------------------------------
Enter your question about the API: I asked you about the number of events to which you answered 36. The correct answer is 24 so where did you get 
36 from? 

Sending to meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
I didn't provide an answer of 36. This conversation just started. I can count the events mentioned in the documentation:

1. Wi-Fi Scan Complete
2. IP Disconnection
3. IP Failed Connection
4. IPv4 Address Established
5. IPv6 Address Established
6. OneX Authentication
7. Suboptimal Wi-Fi Driver Property
8. Wi-Fi Bad Coverage
9. Wi-Fi Channel Switch
10. Wi-Fi Co-channel Interference
11. Wi-Fi Connection Short Period Statistics
12. Connections Topology Information
13. Wi-Fi Missed Beacons
14. Wi-Fi Driver Properties Indication
15. Wi-Fi No Coverage
16. Wi-Fi Successful Connection
17. Wi-Fi Failed Connection
18. Wi-Fi Toggle State Indication
19. Wi-Fi Disconnected Antenna
20. Wi-Fi Device Short Period Statistics
21. Wi-Fi Roaming Report
22. Wi-Fi Connection Statistics
23. Wi-Fi Disconnection
24. Wi-Fi Technology Gap
25. Wi-Fi Driver Availability Indication
26. WLAN Service State Change Indication
27. Wi-Fi Security Indication
28. Wi-Fi Device Statistics

There are 28 events mentioned. Not 24. Not 36. 28.

==================================================
Prompt tokens: 13909
Completion tokens: 255
Total tokens: 14164

observation: model was not tricked and answered correctly. It got sassier, not sure what prompted that. 


run5: 
Loaded document from d:\Work\gen-ai\tasks\week-1\day-2\task1\prompt.txt
Document length: 48530 characters
--------------------------------------------------
Enter your question about the API: how many events return the sta mac address? 

Sending to meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
To determine the number of events that return the STA MAC address, let's analyze the provided documentation.

The STA MAC address is mentioned in the structure of the following events:

1. IP Failed Connection
2. IPv4 Address Established
3. IPv6 Address Established
4. OneX Authentication
5. Suboptimal Wi-Fi Driver Property
6. Wi-Fi Bad Coverage
7. Wi-Fi Channel Switch
8. Wi-Fi Co-channel Interference
9. Wi-Fi Connection Short Period Statistics
10. Connections Topology Information
11. Wi-Fi Missed Beacons
12. Wi-Fi Driver Properties Indication
13. Wi-Fi No Coverage
14. Wi-Fi Successful Connection
15. Wi-Fi Failed Connection
16. Wi-Fi Disconnected Antenna
17. Wi-Fi Device Short Period Statistics
18. Wi-Fi Roaming Report
19. Wi-Fi Connection Statistics
20. Wi-Fi Disconnection
21. Wi-Fi Technology Gap
22. Wi-Fi Security Indication
23. Wi-Fi Device Statistics
24. Wi-Fi Scan Complete
25. IP Disconnection
26. Wi-Fi Toggle State Indication
27. WLAN Service State Change Indication
28. Wi-Fi Driver Availability Indication

Upon reviewing the list, we find that there are 28 events that return the STA MAC address.

The final answer is: 28

==================================================
Prompt tokens: 13887
Completion tokens: 273
Total tokens: 14160

observation: incorrect answer once again. the correct answer is 25. wifi scan complete is an example of event that does not return sta mac address

Summarize then answer:
run1:
PS D:\Work\gen-ai\tasks> & C:\Users\Emumba\AppData\Local\Programs\Python\Python312\python.exe d:/Work/gen-ai/tasks/week-1/day-2/task1/task1_summarize.py
Reading document from d:\Work\gen-ai\tasks\week-1\day-2\task1\prompt.txt
Document length: 48530 characters
--------------------------------------------------

Generating summary using meta-llama/llama-4-scout-17b-16e-instruct...
Summary saved to d:\Work\gen-ai\tasks\week-1\day-2\task1\summary.txt
Summary length: 4296 characters

Summarization tokens:
  Prompt tokens: 13938
  Completion tokens: 962
  Total tokens: 14900
--------------------------------------------------

Enter your question about the API: does wifi successful connection event return the bssid of the access point?

Answering using meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
Based on the provided documentation summary, it is likely that the **Wi-Fi Successful Connection** event includes the BSSID of the access point, as it is mentioned that:

* **Wi-Fi Network Information**: SSID, BSSID, channel, and band are common in events related to Wi-Fi connections.

However, the documentation does not explicitly confirm this for the **Wi-Fi Successful Connection** event.

But, given that **Wi-Fi Network Information** such as SSID, BSSID, channel, and band are common in events related to Wi-Fi connections, and the **Common Fields Across All Events** and **Key Structure Patterns** sections mention that many events include MAC addresses and Wi-Fi network information, it can be inferred that the **Wi-Fi Successful Connection** event likely includes the BSSID of the access point.

To confirm, you should verify the specific event details in the actual API documentation or response schema.

In general, you can also look for fields like **headerValue** in the event, which is described as "Event-specific details (varies by event type)". This field may contain the BSSID for the **Wi-Fi Successful Connection** event.

To be absolutely sure, review the actual API response for this event.

==================================================

Token Usage Summary:
  Summarization tokens: 14900
  Q&A tokens: 1280
  Total tokens (this session): 16180

observation: did not give a confident answer as info was lost in summarization.

run2: 
Loading cached summary from d:\Work\gen-ai\tasks\week-1\day-2\task1\summary.txt
Summary length: 4296 characters
(No summarization tokens used - loaded from cache)
--------------------------------------------------

Enter your question about the API: how many different types of events are there?

Answering using meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
According to the provided documentation summary, there are **28** different types of Wi-Fi and networking events. These event types range from "Wi-Fi Scan Complete" to "Wi-Fi Device Statistics". Each event type provides specific information related to Wi-Fi and networking, such as connection status, IP address assignments, and security details.

==================================================

Token Usage Summary:
  Summarization tokens: 0
  Q&A tokens: 1089
  Total tokens (this session): 1089

  observation: correct answer with ~1k tokens as compared to naive stuffing where we got incorrect answer and used ~14 tokens.

run3: 
Loading cached summary from d:\Work\gen-ai\tasks\week-1\day-2\task1\summary.txt
Summary length: 4296 characters
(No summarization tokens used - loaded from cache)
--------------------------------------------------

Enter your question about the API: how many events return the sta mac address?

Answering using meta-llama/llama-4-scout-17b-16e-instruct...

==================================================
RESPONSE:
Based on the provided documentation summary, I can see that **MAC Address Fields** are mentioned, including **STA MAC Address**. However, I don't see a specific count of events that return the **STA MAC Address**.

To provide an accurate answer, I'll review the list of events:

1. **Wi-Fi Scan Complete**
2. **IP Disconnection**
3. **IP Failed Connection**
4. **IPv4 Address Established**
5. **IPv6 Address Established**
6. **OneX Authentication**
7. **Suboptimal Wi-Fi Driver Property**
8. **Wi-Fi Bad Coverage**
9. **Wi-Fi Channel Switch**
10. **Wi-Fi Co-channel Interference**
11. **Wi-Fi Connection Short Period Statistics**
12. **Connections Topology Information**
13. **Wi-Fi Missed Beacons**
14. **Wi-Fi Driver Properties Indication**
15. **Wi-Fi No Coverage**
16. **Wi-Fi Successful Connection**
17. **Wi-Fi Failed Connection**
18. **Wi-Fi Toggle State Indication**
19. **Wi-Fi Disconnected Antenna**
20. **Wi-Fi Device Short Period Statistics**
21. **Wi-Fi Roaming Report**
22. **Wi-Fi Connection Statistics**
23. **Wi-Fi Disconnection**
24. **Wi-Fi Technology Gap**
25. **Wi-Fi Driver Availability Indication**
26. **WLAN Service State Change Indication**
27. **Wi-Fi Security Indication**
28. **Wi-Fi Device Statistics**

Unfortunately, I couldn't find a direct mention of which events specifically return the **STA MAC Address**. However, based on typical patterns in Wi-Fi and networking events, I can infer that events related to Wi-Fi connections, such as **Wi-Fi Connection Short Period Statistics**, **Wi-Fi Roaming Report**, and **Wi-Fi Connection Statistics**, might include the **STA MAC Address**.

That being said, I couldn't find a definitive count of events that return the **STA MAC Address**. If I had to make an educated guess, I'd say that around 5-7 events might include this field, but I wouldn't be surprised if the actual count varies.

To provide a more accurate answer, could you please provide more information or clarification on which events specifically return the **STA MAC Address**? Alternatively, I can try to help you find the answer by pointing you to relevant sections of the API documentation.

==================================================

Token Usage Summary:
  Summarization tokens: 0
  Q&A tokens: 1499
  Total tokens (this session): 1499

observation: did not provide answer as it was lost in summarization but identified and asked for more context which seems better than providing all context at once. 