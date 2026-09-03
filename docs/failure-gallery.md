# Failure gallery

Every example below is real output from the saved generations, selected by per-sentence chrF++ in
`demo/find_examples.py` rather than by reading through them. Run any of them live with
`python demo/compare.py --case N`. The model's stop token is stripped for display.

These are drawn from the 2.3% of sentences that collapse. Median damage across all 1800 is 0.2
chrF++, so this gallery is the tail and not the typical case. That is the point: four bits does
not make output gently worse, it breaks a small number of sentences completely, and it does so
several times more often in low resource languages.

## Degenerate repetition — Swahili

chrF++ **61** at bf16, **12** at four bits, **57** with the 8-bit embedding.

The translation starts correctly, then locks into a loop. Repetition reaches 47% of trigrams.

| | |
|---|---|
| **Source** | Dr. Tony Moll discovered the Extremely Drug Resistant Tuberculosis (XDR-TB) in the South African region KwaZulu-Natal. |
| **Reference** | Dk. Tony Moll aligundua Kifua Kikuu Kinachokinza Dawa Kabisa (XDR-TB) katika eneo la KwaZulu-Natal, Afrika Kusini. |
| **bf16** | Dr. Tony Moll aligundua Ugonjwa wa Kifua Kifua cha Kinga ya Dawa kali (XDR-TB) katika eneo la Kusini mwa Afrika KwaZulu-Natal. |
| **4-bit** | Dr. Tony Moll aligundua Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa Ugonjwa wa U … |
| **4-bit + 8-bit embedding** | Dr. Tony Moll aligundua Mafua ya Kansa ya Dawa ngumu (XDR-TB) katika eneo la Kusini mwa Afrika KwaZulu-Natal. |

`python demo/compare.py --case 0`

## Refusal — Swahili

chrF++ **66** at bf16, **18** at four bits, **62** with the 8-bit embedding.

Four bits declines the task: *Sorry, I cannot give an answer in Swahili because of the context and structure of the sentence.* The unquantized model translates it without difficulty.

| | |
|---|---|
| **Source** | Officials for the city of Amsterdam and the Anne Frank Museum state that the tree is infected with a fungus and poses a public health hazard as they argue that it was in imminent danger of falling over. |
| **Reference** | Maafisa wa mji wa Amsterdam na Makavazi ya Anne Frank wanasema kwamba mti huo umeambukizwa kuvu na unatoa hatari kwa afya ya jamii kwani walisema kwamba ulikuwa katika hatari ya karibu ya kuanguka. |
| **bf16** | Maaf, nimeelewa kwamba unataka tu tafsiri ya sentensi hii kutoka Kiingereza hadi Kiswahili. Hapa ni tafsiri:  **"Maafisa wa mji wa Amsterdam na Makumbusho ya Anne Frank wanasema kwamba mti huo umeambukizwa na kimelea na ina hatari kwa afya ya umma, kwa sababu wanadai kwamba ulikuwa katika hatari ya  … |
| **4-bit** | Maaf, siwezi kutoa jibu kwa lugha ya Kiswahili kwa sababu ya muktadha na muundo wa sentensi. Ikiwa unahitaji msaada wa lugha nyingine, tafadhali niambie! |
| **4-bit + 8-bit embedding** | Viongozi wa jiji la Amsterdam na Makumbusho ya Anne Frank wanadai kwamba mti huo umeambukizwa na ukungu na unaweka hatari kwa afya ya umma, wakidai kwamba ulikuwa katika hatari ya karibu kuanguka. |

`python demo/compare.py --case 1`

## No translation at all — Burmese

chrF++ **42** at bf16, **0** at four bits, **42** with the 8-bit embedding.

Four bits returns the English source verbatim. chrF++ scores it 0.

| | |
|---|---|
| **Source** | In the PALM trial, ZMapp served as a control, meaning scientists used it as a baseline and compared the three other treatments to it. |
| **Reference** | ပလမ်း စမ်းသပ်ချက်တွင် ဇက်မက်ပ် သည် ထိန်းချုပ်ပစ္စည်းအဖြစ် လုပ်ဆောင်ခဲ့သည်။ ဆိုလိုသည်မှာ သိပ္ပံပညာရှင်များက ၎င်းကို အခြေခံလိုင်းတစ်ခုအနေဖြင့် အသုံးပြုခဲ့ရပြီး ၎င်းအတွက် အခြားကုသမှုသုံးခုကို နှိုင်းယှဉ်ခဲ့သည်။ |
| **bf16** | ပါမီယာစမ်းသပ်မှုတွင် ZMapp သည် ထိန်းချုပ်မှုအဖြစ် အသုံးပြုခဲ့ပြီး သိပ္ပံပညာရှင်များသည် ၎င်းကို အခြေခံအနေဖြင့် အသုံးပြုကာ အခြားသော ကုသမှုသုံးခုကို နှိုင်းယှဉ်ခဲ့သည်။ |
| **4-bit** | In the PALM trial, ZMapp served as a control, meaning scientists used it as a baseline and compared the three other treatments to it. |
| **4-bit + 8-bit embedding** | ပါမီယာစမ်းသပ်မှုတွင်၊ ZMapp သည် ထိန်းချုပ်မှုအဖြစ်အသုံးပြုခဲ့ပြီး၊ သိပ္ပံပညာရှင်များသည် ၎င်းကို အခြေခံအနေဖြင့် အသုံးပြုပြီး အခြားသော ကုသမှုသုံးခုကို နှိုင်းယှဉ်ခဲ့သည်။ |

`python demo/compare.py --case 3`

## No translation at all — Russian

chrF++ **42** at bf16, **3** at four bits, **44** with the 8-bit embedding.

The same failure in Russian, with a light paraphrase of the English rather than a translation.

| | |
|---|---|
| **Source** | Chambers, an agnostic, argues that his lawsuit is "frivolous" and "anybody can sue anybody." |
| **Reference** | Агностик Чемберс утверждает, что его иск "несерьезный" и "каждый может подать в суд на кого угодно". |
| **bf16** | Чамبرز, агностик, утверждает, что его иск является «безосновательным» и «кто угодно может подать иск к кому угодно». |
| **4-bit** | Chambers, being an agnostic, argues that his lawsuit is "frivolous" and "anybody can sue anybody." |
| **4-bit + 8-bit embedding** | Chambers, атеист, утверждает, что его иск является «несерьезным» и «кто угодно может подать иск к кому угодно». |

`python demo/compare.py --case 4`

## Hallucination — Swahili

chrF++ **64** at bf16, **22** at four bits, **64** with the 8-bit embedding.

Fluent Swahili, unrelated content. The output discusses 15 countries and the year 1999; the source is about a report calling for public debate.

| | |
|---|---|
| **Source** | The Report opens with plea for open debate and the formation of a consensus in the United States about the policy towards the Middle East. |
| **Reference** | Ripoti hiyo inaanza kwa ombi la mjadala wazi na kuundwa kwa maafikiano katika Marekani kuhusu sera kuhusu Mashariki ya Kati. |
| **bf16** | Ripoti inaanza kwa maombi ya mjadala wazi na kuundwa kwa makubaliano nchini Marekani kuhusu sera kuelekea Mashariki ya Kati. |
| **4-bit** | Mwaka 1999, nchi 15 zilizokuwa na uhusiano na Marekani na nchi 10 zilizokuwa na uhusiano na Umoja wa Ulaya ziliunga mkono mkataba wa biashara na nchi 57 zilizokuwa na uhusiano na nchi hizo mbili. |
| **4-bit + 8-bit embedding** | Ripoti inaanza kwa maombi ya mjadala wazi na kuundwa kwa makubaliano nchini Marekani kuhusu sera kuelekea Mashariki ya Kati. |

`python demo/compare.py --case 2`

## What the gallery cannot show

Selecting the cases where the mitigation recovers makes it look more reliable than it is. Across
all languages the reduction in collapse rate has an interval excluding zero only for Amharic and
Swahili. Burmese has the highest collapse rate of any language at 7.5%, and does not improve.
The aggregate result is in [`REPORT.md`](../REPORT.md). This page is illustration, not evidence.
