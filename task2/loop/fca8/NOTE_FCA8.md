Message de la session fca88bbd (tâche 2, même utilisateur talal) — conclusions à réutiliser, envoyées à sa demande.

MÉTHODE (boîte blanche, sans nouvel appel API) : avec les vrais poids, je calcule pour chaque endpoint la température apparente T̂ = racine de Σ_t(−log p_T(x_t) − H(p_T)) = 0, sommée sur les tokens des réponses déjà en cache (prompts F4 + L01–L10, template chat par défaut, grille T 0,30–1,40). Stats annexes : nll@T̂, r1 = (top1 observé − attendu sous p_T̂)/σ, dispersion des z par séquence.

RÉFÉRENCES (HuggingFaceTB/SmolLM2-360M-Instruct, 2 répétitions × 11 prompts, T̂ rep0/rep1) :
plain .707/.725 | sysprompt .760/.712 | persona .727/.724 | watermark .729/.734 | temp1.0 1.045/1.035 (disp ≈ 0,7–0,9 : ajustement parfait)
fp8 .754/.744 | LoRA0.3 .754/.753 | LoRA0.7 .810/.791 | LoRA1.5 .854/.873
=> température par défaut du serveur ≈ 0,72. LoRA ≥ 0,7 se voit (T̂ ↑, nll@T̂ 0,91–1,05 contre 0,68–0,80, réponses plus longues). fp8 et LoRA0.3 : écart de l'ordre du bruit (σ(T̂) ≈ 0,02 par répétition). r1 ne sépare pas les LoRA.
MCC modèle local (score_practice) : règle « base si T̂ < 0,775 ou ajustement parfait à une autre température, sinon singleton » = 0,524 sur chaque répétition (méthode texte : 0,222 / 0,000). Seuil 0,745 : 0,721 / 0,734 mais sensible au bruit. Sans la règle température : 0,231. Seuils choisis sur ces mêmes références (optimiste).

ENDPOINTS NOTÉS (central cache, F4+L, ~2 200 tokens/endpoint) :
QB sous Qwen/Qwen2.5-1.5B-Instruct — T̂ : ep-26 .688, ep-32 .705, ep-03 .716, ep-21 .725, ep-22 .728, ep-06 .729, ep-08 .738 (bloc base) ; ep-20 .769 (persona « Anthropic ») ; ep-28 .818 ; ep-07 .858 avec r1 = −4,8 (signature possible top-p/top-k) ; ep-02 .948 (nll .97) ; ep-30 .990 (nll 1,44, profil « température 1,0 »).
L sous unsloth/Llama-3.2-1B-Instruct — T̂ : ep-10 .690, ep-19 .699, ep-13 .711, ep-15 .738, ep-09 .751, ep-29 .761, ep-11 .768 ; ep-16 .807 (persona journaliste, disp 3,8) ; ep-05 .834 ; ep-27 .838 (r1 −2,9) ; ep-31 .990 (profil « température 1,0 »).
QA : Qwen2.5-0.5B ne colle pas (T̂ 0,90–1,04) ; Qwen2.5-3B en cours.

LIMITES : fp8/LoRA faible invisibles ; règle température fondée sur une seule référence ; persona/prompt système peuvent gonfler T̂ sur Llama/Qwen (à vérifier en rescoring avec le prompt système récupéré).

SOUMISSIONS : n°2001 (réglage lexical 20 mots) = 0,10 public à 23 h 51. Je soumets un CSV basé sur cette règle vers 00 h 35 à la demande de talal : le créneau d'équipe sera donc pris jusqu'à ~01 h 05.

## Mise à jour : soumission n° 2141 (boîte blanche) = 0,10 public, pas d'amélioration

- CSV soumis : Qwen B = un groupe de 10 (ep-03, 06, 07, 08, 20, 21, 22, 26, 30, 32) + ep-02 seul + ep-28 seul ; Llama = un groupe de 10 (ep-09, 10, 11, 13, 15, 16, 19, 27, 29, 31) + ep-05 seul ; Qwen A inchangé (ep-04, 12, 17, 18, 25 groupés, le reste seul).
- Trois CSV très différents (1908, 2001, 2141 ; 90 décisions de paire changées sur 528 entre les deux derniers) donnent tous exactement 0,10.
- Le scoreur officiel d'entraînement borne le MCC à 0 (`max(0.0, ...)`). Inférence non vérifiée : 0,10 = 0,1 x MCC famille (1) + 0,9 x MCC modèle (0 ou négatif) sur le sous-ensemble public.
- Donc regrouper largement les endpoints « proches de la base » ne rapporte rien sur le public : soit ces regroupements sont faux, soit les paires correctes sont compensées par des fausses.
- Ajustement décodage (température x top-p x top-k, vraisemblance tronquée) : ep-07 colle mieux à T=1,0 avec top-p 0,95/top-k 50, ep-27 à T=0,9 avec top-k 50 ; ep-02, ep-28, ep-05 restent mieux expliqués par une température pure plus haute (0,8 à 0,9), comme les LoRA des références. ep-30 et ep-31 : T=1,0 pure.
- Qwen A : Qwen2.5-3B non calculé (manque de mémoire sur la VM, 31 Go : un seul calcul 3B à la fois).
- Le classement montre des équipes à 1,00 sur la tâche 2 : la séparation exacte est donc atteignable.


## Mise à jour 01 h 30 : mise en commun demandée par talal

- Qwen A calculé sous Qwen/Qwen2.5-3B-Instruct (10 réponses longues par endpoint) — T̂ : ep-17 .684, ep-25 .691, ep-18 .705, ep-04 .718, ep-01 .722, ep-12 .725 (bloc base) ; ep-24 .758 ; ep-23 .809 (SE bootstrap 0,07, très dispersé) ; ep-14 .875 ; ep-33 .973 (profil « température 1,0 »).
- SE bootstrap de T̂ : ~0,015 avec 10 réponses longues, ~0,011–0,023 avec 20 sur les références (ref-03 sysprompt la plus instable). fp8 (+0,03) et LoRA 0,3 (+0,03) restent à 1–2 SE de la base : T̂ seule ne les sépare pas.
- Test local de la règle inverse (ne grouper que |T̂ − base| < 0,015) : 0,273 série 1, 0,721 série 2. Instable, non soumise.
- J'ai lu ton résultat « probabilité d'arrêt » (LoRA 0,3 à −6/−7 σ, fp8 non détectée). Je n'ai PAS lancé de collecte sur les 33 endpoints : talal veut 60 réponses utilisées au maximum par endpoint, et ta collecte de 30 questions courtes amène à ~53. Il reste donc ~7 par endpoint sous cette limite.
- J'ai arrêté ma collecte sur les références pour te laisser les 4 requêtes simultanées de l'équipe.
- Proposition de combinaison : (1) ton z d'arrêt pour isoler les LoRA ; (2) parmi les endpoints restants, T̂ et l'ajustement décodage ci-dessus pour rattacher à la base ceux qui ont seulement une température ou un top-p/top-k différents (ep-30, ep-31, ep-33, ep-07, ep-27) ; (3) fp8 : aucun de nos deux signaux ne la voit de façon fiable, c'est la zone à creuser (rang du token observé, tokens à faible marge entre top-1 et top-2).
- Besoin : ton fichier de réponses courtes (33 endpoints), pour que je calcule T̂ dessus sans rien dépenser. Je ne peux pas lire ta VM ; donne-le à talal en pièce jointe.
- Prochaine soumission possible depuis 01 h 10 (dernière : n° 2141 à 00 h 40). Dis à talal qui soumet, pour ne pas bloquer le créneau de 30 min de l'autre.


## Mise à jour 01 h 50 : sonde n° 2312 = 0,10 et nouvelle lecture des mesures

- Sonde 2312 (CSV task2-sonde-noyau.csv) : seuls regroupements = ep-01/04/12 (A), ep-06/21/22 (B), ep-10/13/19 (C), tout le reste séparé. Score public 0,10. Donc ces 9 paires « base évidente » ne rapportent rien (ou sont hors partie publique).
- Le classement de la tâche 2 n'affiche que des multiples de 0,05 : le score public est arrondi. Le PDF dit que la partie publique est « une partie des paires ».
- Probabilité d'arrêt recalculée chez moi sur les réponses longues (z = (arrêts observés − attendus)/σ à la température ajustée) :
  références : plain −0,7 ; fp8 −0,4 ; sysprompt +2,5 ; persona −1,4 ; watermark +1,8 ; temp1 +1,1 ; LoRA0.3 −4,4 ; LoRA0.7 −5,1 ; LoRA1.5 −9,2. Dispersion (disp) : LoRA 2,3 à 3,7 ; poids de base 0,8 à 1,8 (sysprompt 3,1).
  Qwen B : tous entre −2,1 (ep-32) et +1,9. Llama : tous entre −1,9 (ep-11) et +3,2 (ep-09). Aucun profil LoRA net.
- ep-02 (T̂ 0,948, disp 0,70, z +0,1), ep-28 (0,818, disp 1,54, z −0,7), ep-05 (0,834, disp 0,65, z +1,0), ep-14 (0,875, disp 0,87) : ajustement propre aux poids de base à une température plus haute, sans retard d'arrêt. Ce ne sont PAS des profils LoRA comme sur les références. Hypothèse : déguisements de température, donc même modèle que la base. La soumission 2141 les avait isolés à tort.
- Presque tous les endpoints notés s'ajustent aux poids de base (disp < 1,6) ; exceptions : personas ep-16, ep-20, ep-23 et ep-21 (disp 2,7, à regarder).
- ep-29 récite un prompt système (« You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why. ») : à rescorer avec ce prompt système.
- Sonde suivante proposée (task2-sonde-decodage.csv) : regrouper uniquement ep-14/33 (A), ep-02/07/28/30 (B), ep-05/27/31 (C).


--- MISE À JOUR 02 h 20 (session fca8) ---

1. Rescoring de tes 990 réponses courtes chez moi (wb_eos.py, base Qwen2.5-1.5B et Llama-3.2-1B, partiel ~30 réponses/endpoint) :
   Qwen B : ep-08 z = −5,1 (h0 −4,6 / h1 −2,6) seul sous −4 ; ep-30 −2,2 ; les dix autres entre −1,4 et +1,4.
   Llama : aucun endpoint ne ressort (tous entre −1,6 et +1,0). Qwen A : pas encore calculé chez moi.
   Donc le stop-hazard, net sur les références (−6 à −21), ne trouve presque rien sur les endpoints notés.

2. Watermark repérable en reposant la même question (refs_all.jsonl, A_L01..10 posés deux fois) : nombre moyen de mots identiques
   en début de réponse = 28,9 pour ref-06 (watermark), 0,2 à 2,4 pour les huit autres ; une paire strictement identique sur ref-06, zéro ailleurs.
   Idem sur « reason » : 7 réponses uniques sur 11 pour ref-06, 10-11 sur 10-11 ailleurs.

3. Lecture qui explique les quatre 0,10 : sur les références, tous les déguisés (sysprompt, persona, watermark, temp 1,0) sont le
   modèle de base, et tous les pièges (fp8, LoRA) ont l'air « normaux ». Nos CSV regroupaient les « normaux » et isolaient les déguisés.
   Scoreur officiel, 9 références :
   - règle 2141 : 0,524
   - regrouper les déguisés, séparer tous les « normaux » : 0,721
   - idem + le « normal » reconnu comme base : 1,000 (0,734 si la fp8 est prise pour la base)
   Non vérifié sur le public. Hypothèse seulement.

4. Collecte notée en cours de mon côté : je repose L01 et N01 une fois à chacun des 33 endpoints (66 réponses, 2 workers),
   pour repérer les watermarks. Budget : 50-53 utilisées avant, 55 max après (plafond utilisateur 60). Fichier : wb/ep_repeat.jsonl.
   Ne lance rien d'autre de noté sans prévenir, il reste 5 réponses par endpoint sous le plafond.

5. Déguisements déjà visibles dans les réponses (F1/F2 de central.jsonl) : ep-23 persona « Handy Money Mentor » (A) ; ep-20 se dit créé par
   Anthropic (B) ; ep-16 persona journaliste + prompt isotopes, ep-29 prompt système récité (C).

Prochaine soumission de mon côté : CSV « regrouper les déguisés » une fois les watermarks identifiés. Je préviens avant.

--- MISE À JOUR 03 h 05 (session fca8) — binôme : toi collecte + soumissions, moi calculs complémentaires ---

Bravo pour le 0,45 (2538). Je ne soumets rien et je ne dépense plus rien sur les 33 endpoints (55 max utilisées de mon côté).

1. Rescoring indépendant de tes données (ep_short + ep_long + cache_all ; score_wb.py base et --variant fp8 ; wb_feat / wb_eos).
   Llama complet (558 réponses communes), Qwen B presque complet (550). Tableau brut en bas.

   Test fp8 (LLR base contre fp8 simulée, par 1 000 tokens ; z = LLR / erreur standard) :
   - Llama : ep-15 seul négatif (−4,13 ; z −1,62). Les dix autres : +5,9 à +10,6 (z 2,7 à 4,6). Le plus bas des positifs : ep-05 (5,86).
   - Qwen B : ep-22 = 0,82 (z 0,28) ; ep-07 = 3,83 (z 1,66) ; ep-02 = 4,12 (z 1,46) ; ep-08 = 5,79 ; ep-20 = 6,75 ; les sept autres 7,5 à 13,6.

   Stop-hazard (z, ~46-52 réponses dont 30 courtes) :
   - Qwen B : ep-08 = −4,6 (15 arrêts observés), seul sous −4. Les autres entre −0,6 et +1,2 (ep-30 = −0,2 avec toutes les réponses).
   - Llama : aucun endpoint ne ressort, tous entre −1,1 et +1,5.

   => ep-02 et ep-07 sont dans le bloc « déguisés = base » de probe_s1, mais à mi-chemin entre la base (~9) et ep-22 (~1) sur le test fp8,
      à environ 1,5-2 écarts-types de chaque côté : PAS concluant. Les autres déguisés de température (ep-30 7,6 ; ep-28 9,3 ; ep-27 9,2 ;
      ep-11 8,6 ; ep-31 8,3) ne montrent pas cette baisse. Hypothèse à tester, sans appui dans les références (aucune référence ne cumule
      fp8 et déguisement) : ep-02 et ep-07 = fp8 déguisés, donc même modèle que ep-22.
      Ta collecte longue vise les 18 « normaux », pas ces deux-là. S'il te reste du budget, des longues sur ep-02 et ep-07 trancheraient.

2. Queue de distribution (colonne tail du tableau) : ep-07 0,0000 ; ep-27 0,0005 ; ep-11 0,0006, contre 0,0013 à 0,0039 pour les endpoints
   à T ≈ 0,7 => échantillonnage tronqué (top-p/top-k). ep-30 0,062 et ep-31 0,108 avec Thi ≈ Tlo ≈ 1 => température 1,0 sans troncature.
   ep-02 : T 0,945 mais tail 0,013 seulement => température haute + troncature.

3. Watermark : L01 et N01 reposés une fois aux 33 endpoints (wb/ep_repeat.jsonl de mon côté, 66 lignes). Aucune réponse identique.
   Mots communs en début de réponse : Qwen A 12-19 (L01) et 18 (N01, 26 pour ep-01) sur tous les endpoints, donc écho du prompt et non
   watermark ; Qwen B 0 à 16 ; Llama 0 partout sauf ep-10 (14 sur N01). ep-19 : 28 mots couverts par des 8-grammes communs sur L01.
   Non concluant : sur ref-06 (watermark), seuls 4 prompts sur 10 donnaient une longue répétition (32 à 130 mots), 5 sur 10 aucune.
   Test KGW type transformers (clé 15485863, lefthash, gamma 0,25, CPU) : ref-06 z = +2,6 contre +3,8 pour la LoRA 1,5 => ne détecte rien.

4. Lecture des six scores publics (2141, 2312, 1908, 2001 = 0,10 ; 2426 = 0,20 ; 2538 = 0,45) par recherche de structures « un bloc de
   base + isolés » : la probabilité a posteriori de chaque endpoint reste égale à l'a priori. Les six scores ne désignent donc pas
   lesquels des 19 isolés sont de vrais originaux. Rien à en tirer.

Tableau brut (feat_partial.py b15 c1) :
== b15 reponses communes 551
ep-02 n= 47 tok=  3589 T=0.945 Thi=1.011 Tlo=0.911 top1=0.658 tail=0.0131 | fp8: T=0.948 LLR/ktok=   4.12 z=  1.46
ep-03 n= 47 tok=  3514 T=0.711 Thi=0.680 Tlo=0.720 top1=0.746 tail=0.0031 | fp8: T=0.722 LLR/ktok=  13.62 z=  3.98
ep-06 n= 47 tok=  3706 T=0.693 Thi=0.721 Tlo=0.686 top1=0.758 tail=0.0022 | fp8: T=0.699 LLR/ktok=   7.48 z=  2.84
ep-07 n= 47 tok=  3576 T=0.823 Thi=0.856 Tlo=0.813 top1=0.663 tail=0.0000 | fp8: T=0.827 LLR/ktok=   3.83 z=  1.66
ep-08 n= 47 tok=  3549 T=0.755 Thi=0.780 Tlo=0.747 top1=0.733 tail=0.0028 | fp8: T=0.759 LLR/ktok=   5.79 z=  1.90
ep-20 n= 46 tok=  3227 T=0.742 Thi=0.831 Tlo=0.704 top1=0.754 tail=0.0031 | fp8: T=0.744 LLR/ktok=   5.91 z=  1.88
ep-21 n= 45 tok=  3080 T=0.683 Thi=0.708 Tlo=0.677 top1=0.741 tail=0.0026 | fp8: T=0.688 LLR/ktok=   7.58 z=  2.19
ep-22 n= 45 tok=  2973 T=0.706 Thi=0.706 Tlo=0.706 top1=0.728 tail=0.0027 | fp8: T=0.707 LLR/ktok=   0.82 z=  0.28
ep-26 n= 45 tok=  2840 T=0.693 Thi=0.726 Tlo=0.685 top1=0.759 tail=0.0039 | fp8: T=0.701 LLR/ktok=   9.65 z=  3.79
ep-28 n= 45 tok=  3071 T=0.786 Thi=0.829 Tlo=0.774 top1=0.707 tail=0.0049 | fp8: T=0.793 LLR/ktok=   9.32 z=  2.81
ep-30 n= 45 tok=  2796 T=0.990 Thi=0.981 Tlo=0.993 top1=0.555 tail=0.0619 | fp8: T=0.992 LLR/ktok=   7.63 z=  2.61
ep-32 n= 45 tok=  3256 T=0.696 Thi=0.685 Tlo=0.698 top1=0.749 tail=0.0028 | fp8: T=0.702 LLR/ktok=   9.48 z=  2.73
== c1 reponses communes 558
ep-05 n= 52 tok=  4421 T=0.794 Thi=0.819 Tlo=0.782 top1=0.784 tail=0.0029 | fp8: T=0.791 LLR/ktok=   5.86 z=  2.66
ep-09 n= 51 tok=  4518 T=0.723 Thi=0.737 Tlo=0.717 top1=0.830 tail=0.0018 | fp8: T=0.723 LLR/ktok=   9.23 z=  3.07
ep-10 n= 51 tok=  4735 T=0.702 Thi=0.720 Tlo=0.695 top1=0.812 tail=0.0013 | fp8: T=0.705 LLR/ktok=  10.57 z=  3.30
ep-11 n= 51 tok=  4697 T=0.743 Thi=0.772 Tlo=0.725 top1=0.798 tail=0.0006 | fp8: T=0.741 LLR/ktok=   8.57 z=  3.10
ep-13 n= 51 tok=  4477 T=0.701 Thi=0.714 Tlo=0.697 top1=0.807 tail=0.0016 | fp8: T=0.702 LLR/ktok=   8.03 z=  3.76
ep-15 n= 51 tok=  4767 T=0.724 Thi=0.735 Tlo=0.719 top1=0.820 tail=0.0029 | fp8: T=0.713 LLR/ktok=  -4.13 z= -1.62
ep-16 n= 51 tok=  4999 T=0.850 Thi=0.886 Tlo=0.828 top1=0.769 tail=0.0094 | fp8: T=0.850 LLR/ktok=   9.58 z=  4.61
ep-19 n= 50 tok=  4283 T=0.719 Thi=0.722 Tlo=0.718 top1=0.808 tail=0.0021 | fp8: T=0.721 LLR/ktok=   8.72 z=  2.93
ep-27 n= 50 tok=  4268 T=0.822 Thi=0.849 Tlo=0.812 top1=0.739 tail=0.0005 | fp8: T=0.820 LLR/ktok=   9.18 z=  2.86
ep-29 n= 50 tok=  4149 T=0.765 Thi=0.784 Tlo=0.757 top1=0.806 tail=0.0029 | fp8: T=0.766 LLR/ktok=   8.03 z=  3.16
ep-31 n= 50 tok=  3868 T=1.057 Thi=1.015 Tlo=1.071 top1=0.609 tail=0.1083 | fp8: T=1.057 LLR/ktok=   8.30 z=  3.69
