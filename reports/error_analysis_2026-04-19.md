# Error Analysis Report

- Date: `2026-04-19`
- Model: `D:/CV-Analyzer-AI-main/saved_model/model.pkl`
- Dataset: `D:/CV-Analyzer-AI-main/clean_resume_data.csv` (rows used: `2483`)
- Holdout split: test_size=`0.2`, random_state=`42`

## Summary Metrics

- Accuracy: `79.68%`
- Macro F1: `0.743`
- Weighted F1: `0.785`

## Top Confusions (True -> Predicted)

1. `ARTS` -> `TEACHER`: `4` (19.0% of `ARTS` test samples)
   - "language arts teacher professional summary continue working children well youth able utilize skills expertise area elementary middle school core qualifications ability communicate inspire trust confidence motivate children well understand children education..."
   - "eighth ninth tenth english teacher summary motivated literacy specialist reading la teacher extensive knowledge education system educational testing standards exceptional communicator advanced problem solving skills versed working well learning styles effec..."
   - "bilingual language arts sixth grade teacher summary dedicated enthusiastic professional four years experience education proven expertise establishing rapport building trust students parents administrators community members possess strong communication skill..."
2. `ARTS` -> `ADVOCATE`: `3` (14.3% of `ARTS` test samples)
   - "expressive arts program leader professional experience expressive arts program leader city state formulated proposed authorized implement arts project weekly therapeutic expressive arts program risk cps middle school students residing low income urban neigh..."
   - "inside account manager summary account manager eight years experience shi fill sales team manager position strong organizational analytical problem resolution skills vast account knowledge highlights team leadership customer service expert experienced volum..."
   - "school counselor summary seeking full time counseling psychology instructor position help students become successful life design healthy learning social environments assisting develop educational plan promoting multiculturalism school activities help testin..."
3. `SALES` -> `APPAREL`: `2` (8.7% of `SALES` test samples)
   - "sales manager summary friendly enthusiastic six years specialization hospitality able learn new tasks quickly proficient growing key customer relationships represent establishment friendly professional demeanor times able work fast paced establishment passi..."
   - "sales summary years sales operations management experience specialty big box retail years sales experience automotive sector experienced hiring training supervision coaching proven skills operations human resource management planning negotiating organizing ..."
4. `PUBLIC-RELATIONS` -> `FITNESS`: `2` (9.1% of `PUBLIC-RELATIONS` test samples)
   - "bartender experience bartender company name city state marketing public relations years maintain proper adequate set bar daily basis responsible maintaining stock preparing storing garnishes juices perishables ensure product quality attend stand ups prior f..."
   - "billing services associate professional summary detail oriented versatile government commercial non profit billing professional proven ability efficiently manage multiple assignments meeting strict deadlines excels cultivating managing internal external cli..."
5. `ENGINEERING` -> `INFORMATION-TECHNOLOGY`: `2` (8.3% of `ENGINEERING` test samples)
   - "software engineering team lead years experience field infrastructure consulting operations worked various microsoft technologies system center suite scom sccm scvmm virtualization hyper v windows active directory dns dhcp windows clusters scripting powershe..."
   - "corporate engineering support technician summary technical support engineer technical support telecom support networking support software pc lan troubleshooting accomplished technical support years experience troubleshooting maintaining user applications wo..."
6. `DIGITAL-MEDIA` -> `INFORMATION-TECHNOLOGY`: `2` (10.5% of `DIGITAL-MEDIA` test samples)
   - "software engineer summary accomplished development professional ten years experience integrating operations processes sustainable customized applications promote team performance efficiency gains apt student programming markup languages matching multiple te..."
   - "sprint isp management vendor qualifications windows xp windows windows nt red hat limited ubuntu limited virtualization technology esx esxi ms hyperv security systems load balancers brocade ssl load balancer f bigip aft ltm load balancer ssl cert management..."
7. `AUTOMOBILE` -> `ADVOCATE`: `2` (28.6% of `AUTOMOBILE` test samples)
   - "legal assistant summary dedicated focused administrative assistant excels prioritizing completing multiple tasks simultaneously following achieve project goals seeking role increased responsibility authority highlights fluent spanish microsoft office profic..."
   - "software support specialist professional summary dedicated customer service representative motivated maintain customer satisfaction contribute company success skill highlights strong organizational skills energetic work attitude telephone inquiries speciali..."
8. `ARTS` -> `HEALTHCARE`: `2` (9.5% of `ARTS` test samples)
   - "college assistant summary professional leader strong emphasis management initiatives focused developing comprehensive family child related programs services goal oriented professional strong leadership capabilities detail oriented exhibiting excellent commu..."
   - "medical billing specialist skill highlights microsoft office products word excel access powerpoint type words minute experience face face customer service interaction experience communications external clients email conference calling create maintain detail..."
9. `APPAREL` -> `SALES`: `2` (10.5% of `APPAREL` test samples)
   - "sales consultant summary talented sales professional effectively multi tasks balances customer needs company demands efficiently builds loyalty long term relationships customers consistently reaching sales targets accomplishments golden eagle award winner g..."
   - "embroidery machine operator summary find employment good reputable company chance advance best abilities job skills skills customer service cashier pharmacy sales associate embroidery machine operator sewing machine operator telephone fax mahcines filing of..."
10. `APPAREL` -> `PUBLIC-RELATIONS`: `2` (10.5% of `APPAREL` test samples)
   - "ceo president executive profile senior marketing executive experienced music sports entertainment industry ability build market presence track record accelerating growth creating executing integrated marketing strategic marketing programs improved visibilit..."
   - "director pr social media executive dynamic results driven senior public relations executive years experience impacting brand presence performance profitability internationally strategic leader notable success development execution public relations marketing..."
11. `AGRICULTURE` -> `ENGINEERING`: `2` (15.4% of `AGRICULTURE` test samples)
   - "driver summary biological science technician years experience vineyard property maintenance customer service experience well experience company name july current driver city state company name february june biological science technician city state caretaker..."
   - "safety intern areas expertise occupational safety microsoft word excel power point e tools professional experience safety intern company name city state walkthroughs plant make sure proper personal protective equipment worn employees transition material saf..."
12. `ADVOCATE` -> `HEALTHCARE`: `2` (8.3% of `ADVOCATE` test samples)
   - "practicum experience summary nurse practitioner focused providing quality care maintaining direct lines communication patients entire health care team superb interpersonal leadership skills enjoys challenges learning new concepts skill sets certifications r..."
   - "patient experience manager summary results oriented manager thrives fast paced competitive environments brings strong presentation analytical problem solving skills systematically savvy management sales customer service individual multiple leadership experi..."
13. `TEACHER` -> `HEALTHCARE`: `1` (5.0% of `TEACHER` test samples)
   - "substitute teacher summary health administration hospital operations public service major intelligent enthusiastic young professional progressive career healthcare administration looking outstanding opportunities experience working various industries capabl..."
14. `SALES` -> `CONSTRUCTION`: `1` (4.3% of `SALES` test samples)
   - "sales professional summary talented construction manager twenty years success various projects independent contractor solid experience managing levels small large scale projects extensive experience preparation complete cost estimation accomplished completi..."
15. `SALES` -> `BUSINESS-DEVELOPMENT`: `1` (4.3% of `SALES` test samples)
   - "sales professional summary experienced manager excellent client project management skills action oriented strong ability communicate effectively technology executive business audiences analyst extensive experience microsoft office proficiencies include micr..."

## Full Classification Report

```
                        precision    recall  f1-score   support

            ACCOUNTANT      0.920     0.958     0.939        24
              ADVOCATE      0.690     0.833     0.755        24
           AGRICULTURE      1.000     0.615     0.762        13
               APPAREL      0.600     0.316     0.414        19
                  ARTS      0.667     0.381     0.485        21
            AUTOMOBILE      1.000     0.286     0.444         7
              AVIATION      0.875     0.875     0.875        24
               BANKING      0.792     0.826     0.809        23
                   BPO      0.000     0.000     0.000         4
  BUSINESS-DEVELOPMENT      0.840     0.875     0.857        24
                  CHEF      0.950     0.792     0.864        24
          CONSTRUCTION      0.818     0.818     0.818        22
            CONSULTANT      0.920     1.000     0.958        23
              DESIGNER      0.905     0.905     0.905        21
         DIGITAL-MEDIA      0.684     0.684     0.684        19
           ENGINEERING      0.800     0.833     0.816        24
               FINANCE      0.852     0.958     0.902        24
               FITNESS      0.680     0.739     0.708        23
            HEALTHCARE      0.680     0.739     0.708        23
                    HR      0.875     0.955     0.913        22
INFORMATION-TECHNOLOGY      0.800     1.000     0.889        24
      PUBLIC-RELATIONS      0.727     0.727     0.727        22
                 SALES      0.792     0.826     0.809        23
               TEACHER      0.679     0.950     0.792        20

              accuracy                          0.797       497
             macro avg      0.773     0.746     0.743       497
          weighted avg      0.794     0.797     0.785       497

```
