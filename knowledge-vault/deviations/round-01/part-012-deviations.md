---
deviation_count: 10
part_id: part_012
pov: sendai
round: 1
---

# Deviations: part_012, Round 01

## Deviation 1

- **Category:** attribution
- **Severity:** major
- **POV-specific:** false
- **JP source:** 彼氏が欲しい。
格好良くて、浮気しない彼氏がいい。
彼氏が、彼氏が、彼氏が。
- **LLM rendering:** I want a boyfriend.
A cool, good-looking boyfriend who wouldn't cheat on me.
Boyfriend, boyfriend, boyfriend.
- **Reference rendering:** “I want a boyfriend.”
“I want a cool boyfriend who won’t cheat on me.”
Boyfriend, boyfriend, boyfriend.
- **Notes:** Leaving the first two lines unquoted makes them read like Sendai’s own narration rather than Umina’s repeated spoken lines.

## Deviation 2

- **Category:** attribution
- **Severity:** major
- **POV-specific:** false
- **JP source:** 羽美奈が私の名前を呼んで、笑顔を作る。
- **LLM rendering:** Umina called out my name as she put on a smile.
- **Reference rendering:** I forced a smile on my face when I heard Umina say my name.
- **Violates rule:** [[rule-000-04]]
- **Notes:** The LLM carries Umina over as the subject of 笑顔を作る, while the reference resolves the omitted main-clause subject as the narrator.

## Deviation 3

- **Category:** formatting
- **Severity:** minor
- **POV-specific:** false
- **JP source:** 「羽美奈の方がモテる」
- **LLM rendering:** the conclusion of, "Umina, you're way more popular than I am."
- **Reference rendering:** 「You’re more popular, Umina」
- **Violates rule:** [[rule-000-03]]
- **Notes:** The LLM replaces the source-style Japanese brackets with English quotation marks for a mid-narration quoted line.

## Deviation 4

- **Category:** translationese
- **Severity:** minor
- **POV-specific:** false
- **JP source:** 呼び出しのメッセージ
- **LLM rendering:** her summons
- **Reference rendering:** she messaged me
- **Violates rule:** [[rule-000-06]]
- **Notes:** Using the noun “summons” for a message/call-over is stiff and unnatural in contemporary narration, and the LLM repeats this register elsewhere with 呼び出す.

## Deviation 5

- **Category:** translationese
- **Severity:** major
- **POV-specific:** false
- **JP source:** 家主
- **LLM rendering:** the owner of the house
- **Reference rendering:** the girl who actually lived here
- **Violates rule:** [[rule-000-06]]
- **Notes:** “The owner of the house” misleadingly implies property ownership rather than the person whose room/apartment this is.

## Deviation 6

- **Category:** style-rhythm
- **Severity:** major
- **POV-specific:** false
- **JP source:** そう素っ気なく言うと、宮城がファンヒーターで暑いくらいに暖められた部屋を出てキッチンへ向かう。
- **LLM rendering:** After saying that curtly, Miyagi left the room – which was hot enough to be uncomfortable thanks to the space heater – and headed for the kitchen.
- **Reference rendering:** Miyagi said bluntly as she went to turn her space heater on, raising the setting high enough to turn the whole room hot. Then she walked out of the room and headed for the kitchen.
- **Violates rule:** [[rule-000-07]]
- **Notes:** The LLM uses the exact terse “After saying that…” pattern the active rule warns against, stripping the cushioned action beat and reference rhythm.

## Deviation 7

- **Category:** style-rhythm
- **Severity:** minor
- **POV-specific:** false
- **JP source:** まな板と包丁を洗いながら問いかけると、低くて小さな声が聞こえてきた。
- **LLM rendering:** When I asked her that as I washed the cutting board and knife, I heard a low, quiet voice.
- **Reference rendering:** I asked her while washing the cutting board. Then, I heard her respond in a low voice.
- **Violates rule:** [[rule-000-07]]
- **Notes:** The LLM collapses a beat-and-response structure into a single subordinate clause, matching the translationese pattern called out in the active rule.

## Deviation 8

- **Category:** sentence-structure
- **Severity:** minor
- **POV-specific:** false
- **JP source:** 水道が止められた。
- **LLM rendering:** the water was turned off.
- **Reference rendering:** she immediately turned the faucet off.
- **Violates rule:** [[rule-000-06]]
- **Notes:** The English passive calques the Japanese construction and obscures Miyagi’s immediate agency in the action.

## Deviation 9

- **Category:** translationese
- **Severity:** minor
- **POV-specific:** false
- **JP source:** 切った指
- **LLM rendering:** the cut finger
- **Reference rendering:** the cut on my finger
- **Violates rule:** [[rule-000-06]]
- **Notes:** “The cut finger” is an unnatural noun-phrase calque where fluent English would specify the cut on her finger.

## Deviation 10

- **Category:** style-rhythm
- **Severity:** minor
- **POV-specific:** false
- **JP source:** 流水で洗って、絆創膏を貼って。
キッチンから宮城を追い出す。
- **LLM rendering:** Rinse it under running water, slap on a bandage, and kick Miyagi out of the kitchen.
- **Reference rendering:** Wash the wound and then stick a bandage on it.
And then kick Miyagi out of the kitchen.
- **Violates rule:** [[rule-000-07]]
- **Notes:** The LLM mashes two short beat sentences into one punchier list, losing the reference’s deliberate paragraph rhythm.