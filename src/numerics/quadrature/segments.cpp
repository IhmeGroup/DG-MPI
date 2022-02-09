#include "numerics/quadrature/segment.h"
#include "numerics/nodes.h"
#include <math.h>

namespace SegmentQuadrature {


KOKKOS_FUNCTION
void get_segment_weights_gl(const int order, 
    View<rtype*>::HostMirror& wts){    
    switch(order){
        case 0 :
        case 1 :
        	resize(wts, 1);
            wts(0) = (2.0000000000000000000000000000000);
            break;
        case 2 :
        case 3 :
        	resize(wts, 2);
            wts(0) = (1.0000000000000000000000000000000);
            wts(1) = (1.0000000000000000000000000000000);
            break;
        case 4 :
        case 5 :
        	resize(wts, 3);
            wts(0) = (0.55555555555555555555555555555556);
            wts(1) = (0.88888888888888888888888888888889);
            wts(2) = (0.55555555555555555555555555555556);
            break;
        case 6 :
        case 7 :
        	resize(wts, 4);
            wts(0) = (0.34785484513745385737306394922200);
            wts(1) = (0.65214515486254614262693605077800);
            wts(2) = (0.65214515486254614262693605077800);
            wts(3) = (0.34785484513745385737306394922200);
            break;
        case 8 :
        case 9 :
        	resize(wts, 5);
            wts(0) = (0.23692688505618908751426404071992);
            wts(1) = (0.47862867049936646804129151483564);
            wts(2) = (0.56888888888888888888888888888889);
            wts(3) = (0.47862867049936646804129151483564);
            wts(4) = (0.23692688505618908751426404071992);
            break;
        case 10 :
        case 11 :
        	resize(wts, 6);
            wts(0) = (0.17132449237917034504029614217273);
            wts(1) = (0.36076157304813860756983351383772);
            wts(2) = (0.46791393457269104738987034398955);
            wts(3) = (0.46791393457269104738987034398955);
            wts(4) = (0.36076157304813860756983351383772);
            wts(5) = (0.17132449237917034504029614217273);
            break;
        case 12 :
        case 13 :
        	resize(wts, 7);
            wts(0) = (0.12948496616886969327061143267908);
            wts(1) = (0.27970539148927666790146777142378);
            wts(2) = (0.38183005050511894495036977548898);
            wts(3) = (0.41795918367346938775510204081633);
            wts(4) = (0.38183005050511894495036977548898);
            wts(5) = (0.27970539148927666790146777142378);
            wts(6) = (0.12948496616886969327061143267908);
            break;
        case 14 :
        case 15 :
        	resize(wts, 8);
            wts(0) = (0.10122853629037625915253135430996);
            wts(1) = (0.22238103445337447054435599442624);
            wts(2) = (0.31370664587788728733796220198660);
            wts(3) = (0.36268378337836198296515044927720);
            wts(4) = (0.36268378337836198296515044927720);
            wts(5) = (0.31370664587788728733796220198660);
            wts(6) = (0.22238103445337447054435599442624);
            wts(7) = (0.10122853629037625915253135430996);
            break;
        case 16 :
        case 17 :
        	resize(wts, 9);
            wts(0) = (0.081274388361574411971892158110524);
            wts(1) = (0.18064816069485740405847203124291);
            wts(2) = (0.26061069640293546231874286941863);
            wts(3) = (0.31234707704000284006863040658444);
            wts(4) = (0.33023935500125976316452506928697);
            wts(5) = (0.31234707704000284006863040658444);
            wts(6) = (0.26061069640293546231874286941863);
            wts(7) = (0.18064816069485740405847203124291);
            wts(8) = (0.081274388361574411971892158110524);
            break;
        case 18 :
        case 19 :
        	resize(wts, 10);
            wts(0) = (0.066671344308688137593568809893332);
            wts(1) = (0.14945134915058059314577633965770);
            wts(2) = (0.21908636251598204399553493422816);
            wts(3) = (0.26926671930999635509122692156947);
            wts(4) = (0.29552422471475287017389299465134);
            wts(5) = (0.29552422471475287017389299465134);
            wts(6) = (0.26926671930999635509122692156947);
            wts(7) = (0.21908636251598204399553493422816);
            wts(8) = (0.14945134915058059314577633965770);
            wts(9) = (0.066671344308688137593568809893332);
            break;
        case 20 :
        case 21 :
        	resize(wts, 11);
            wts(0) = (0.055668567116173666482753720442549);
            wts(1) = (0.12558036946490462463469429922394);
            wts(2) = (0.18629021092773425142609764143166);
            wts(3) = (0.23319376459199047991852370484318);
            wts(4) = (0.26280454451024666218068886989051);
            wts(5) = (0.27292508677790063071448352833634);
            wts(6) = (0.26280454451024666218068886989051);
            wts(7) = (0.23319376459199047991852370484318);
            wts(8) = (0.18629021092773425142609764143166);
            wts(9) = (0.12558036946490462463469429922394);
            wts(10) = (0.055668567116173666482753720442549);
            break;
        case 22 :
        case 23 :
        	resize(wts, 12);
            wts(0) = (0.047175336386511827194615961485017);
            wts(1) = (0.10693932599531843096025471819400);
            wts(2) = (0.16007832854334622633465252954336);
            wts(3) = (0.20316742672306592174906445580980);
            wts(4) = (0.23349253653835480876084989892488);
            wts(5) = (0.24914704581340278500056243604295);
            wts(6) = (0.24914704581340278500056243604295);
            wts(7) = (0.23349253653835480876084989892488);
            wts(8) = (0.20316742672306592174906445580980);
            wts(9) = (0.16007832854334622633465252954336);
            wts(10) = (0.10693932599531843096025471819400);
            wts(11) = (0.047175336386511827194615961485017);
            break;
        case 24 :
        case 25 :
        	resize(wts, 13);
            wts(0) = (0.040484004765315879520021592200986);
            wts(1) = (0.092121499837728447914421775953797);
            wts(2) = (0.13887351021978723846360177686887);
            wts(3) = (0.17814598076194573828004669199610);
            wts(4) = (0.20781604753688850231252321930605);
            wts(5) = (0.22628318026289723841209018603978);
            wts(6) = (0.23255155323087391019458951526884);
            wts(7) = (0.22628318026289723841209018603978);
            wts(8) = (0.20781604753688850231252321930605);
            wts(9) = (0.17814598076194573828004669199610);
            wts(10) = (0.13887351021978723846360177686887);
            wts(11) = (0.092121499837728447914421775953797);
            wts(12) = (0.040484004765315879520021592200986);
            break;
        case 26 :
        case 27 :
        	resize(wts, 14);
            wts(0) = (0.035119460331751863031832876138192);
            wts(1) = (0.080158087159760209805633277062854);
            wts(2) = (0.12151857068790318468941480907248);
            wts(3) = (0.15720316715819353456960193862384);
            wts(4) = (0.18553839747793781374171659012516);
            wts(5) = (0.20519846372129560396592406566122);
            wts(6) = (0.21526385346315779019587644331626);
            wts(7) = (0.21526385346315779019587644331626);
            wts(8) = (0.20519846372129560396592406566122);
            wts(9) = (0.18553839747793781374171659012516);
            wts(10) = (0.15720316715819353456960193862384);
            wts(11) = (0.12151857068790318468941480907248);
            wts(12) = (0.080158087159760209805633277062854);
            wts(13) = (0.035119460331751863031832876138192);
            break;
        case 28 :
        case 29 :
        	resize(wts, 15);
            wts(0) = (0.030753241996117268354628393577204);
            wts(1) = (0.070366047488108124709267416450667);
            wts(2) = (0.10715922046717193501186954668587);
            wts(3) = (0.13957067792615431444780479451103);
            wts(4) = (0.16626920581699393355320086048121);
            wts(5) = (0.18616100001556221102680056186642);
            wts(6) = (0.19843148532711157645611832644384);
            wts(7) = (0.20257824192556127288062019996752);
            wts(8) = (0.19843148532711157645611832644384);
            wts(9) = (0.18616100001556221102680056186642);
            wts(10) = (0.16626920581699393355320086048121);
            wts(11) = (0.13957067792615431444780479451103);
            wts(12) = (0.10715922046717193501186954668587);
            wts(13) = (0.070366047488108124709267416450667);
            wts(14) = (0.030753241996117268354628393577204);
            break;
        case 30 :
        case 31 :
        	resize(wts, 16);
            wts(0) = (0.027152459411754094851780572456018);
            wts(1) = (0.062253523938647892862843836994378);
            wts(2) = (0.095158511682492784809925107602246);
            wts(3) = (0.12462897125553387205247628219202);
            wts(4) = (0.14959598881657673208150173054748);
            wts(5) = (0.16915651939500253818931207903036);
            wts(6) = (0.18260341504492358886676366796922);
            wts(7) = (0.18945061045506849628539672320828);
            wts(8) = (0.18945061045506849628539672320828);
            wts(9) = (0.18260341504492358886676366796922);
            wts(10) = (0.16915651939500253818931207903036);
            wts(11) = (0.14959598881657673208150173054748);
            wts(12) = (0.12462897125553387205247628219202);
            wts(13) = (0.095158511682492784809925107602246);
            wts(14) = (0.062253523938647892862843836994378);
            wts(15) = (0.027152459411754094851780572456018);
            break;
        case 32 :
        case 33 :
        	resize(wts, 17);
            wts(0) = (0.024148302868547931960110026287565);
            wts(1) = (0.055459529373987201129440165358245);
            wts(2) = (0.085036148317179180883535370191062);
            wts(3) = (0.11188384719340397109478838562636);
            wts(4) = (0.13513636846852547328631998170235);
            wts(5) = (0.15404576107681028808143159480196);
            wts(6) = (0.16800410215645004450997066378832);
            wts(7) = (0.17656270536699264632527099011320);
            wts(8) = (0.17944647035620652545826564426189);
            wts(9) = (0.17656270536699264632527099011320);
            wts(10) = (0.16800410215645004450997066378832);
            wts(11) = (0.15404576107681028808143159480196);
            wts(12) = (0.13513636846852547328631998170235);
            wts(13) = (0.11188384719340397109478838562636);
            wts(14) = (0.085036148317179180883535370191062);
            wts(15) = (0.055459529373987201129440165358245);
            wts(16) = (0.024148302868547931960110026287565);
            break;
        case 34 :
        case 35 :
        	resize(wts, 18);
            wts(0) = (0.021616013526483310313342710266452);
            wts(1) = (0.049714548894969796453334946202639);
            wts(2) = (0.076425730254889056529129677616637);
            wts(3) = (0.10094204410628716556281398492483);
            wts(4) = (0.12255520671147846018451912680020);
            wts(5) = (0.14064291467065065120473130375195);
            wts(6) = (0.15468467512626524492541800383637);
            wts(7) = (0.16427648374583272298605377646593);
            wts(8) = (0.16914238296314359184065647013499);
            wts(9) = (0.16914238296314359184065647013499);
            wts(10) = (0.16427648374583272298605377646593);
            wts(11) = (0.15468467512626524492541800383637);
            wts(12) = (0.14064291467065065120473130375195);
            wts(13) = (0.12255520671147846018451912680020);
            wts(14) = (0.10094204410628716556281398492483);
            wts(15) = (0.076425730254889056529129677616637);
            wts(16) = (0.049714548894969796453334946202639);
            wts(17) = (0.021616013526483310313342710266452);
            break;
        case 36 :
        case 37 :
        	resize(wts, 19);
            wts(0) = (0.019461788229726477036312041464438);
            wts(1) = (0.044814226765699600332838157401994);
            wts(2) = (0.069044542737641226580708258006013);
            wts(3) = (0.091490021622449999464462094123840);
            wts(4) = (0.11156664554733399471602390168177);
            wts(5) = (0.12875396253933622767551578485688);
            wts(6) = (0.14260670217360661177574610944190);
            wts(7) = (0.15276604206585966677885540089766);
            wts(8) = (0.15896884339395434764995643946505);
            wts(9) = (0.16105444984878369597916362532092);
            wts(10) = (0.15896884339395434764995643946505);
            wts(11) = (0.15276604206585966677885540089766);
            wts(12) = (0.14260670217360661177574610944190);
            wts(13) = (0.12875396253933622767551578485688);
            wts(14) = (0.11156664554733399471602390168177);
            wts(15) = (0.091490021622449999464462094123840);
            wts(16) = (0.069044542737641226580708258006013);
            wts(17) = (0.044814226765699600332838157401994);
            wts(18) = (0.019461788229726477036312041464438);
            break;
        case 38 :
        case 39 :
        	resize(wts, 20);
            wts(0) = (0.017614007139152118311861962351853);
            wts(1) = (0.040601429800386941331039952274932);
            wts(2) = (0.062672048334109063569506535187042);
            wts(3) = (0.083276741576704748724758143222046);
            wts(4) = (0.10193011981724043503675013548035);
            wts(5) = (0.11819453196151841731237737771138);
            wts(6) = (0.13168863844917662689849449974816);
            wts(7) = (0.14209610931838205132929832506716);
            wts(8) = (0.14917298647260374678782873700197);
            wts(9) = (0.15275338713072585069808433195510);
            wts(10) = (0.15275338713072585069808433195510);
            wts(11) = (0.14917298647260374678782873700197);
            wts(12) = (0.14209610931838205132929832506716);
            wts(13) = (0.13168863844917662689849449974816);
            wts(14) = (0.11819453196151841731237737771138);
            wts(15) = (0.10193011981724043503675013548035);
            wts(16) = (0.083276741576704748724758143222046);
            wts(17) = (0.062672048334109063569506535187042);
            wts(18) = (0.040601429800386941331039952274932);
            wts(19) = (0.017614007139152118311861962351853);
            break;
        case 40 :
        case 41 :
        	resize(wts, 21);
            wts(0) = (0.016017228257774333324224616858471);
            wts(1) = (0.036953789770852493799950668299330);
            wts(2) = (0.057134425426857208283635826472448);
            wts(3) = (0.076100113628379302017051653300183);
            wts(4) = (0.093444423456033861553289741113932);
            wts(5) = (0.10879729916714837766347457807011);
            wts(6) = (0.12183141605372853419536717712573);
            wts(7) = (0.13226893863333746178105257449678);
            wts(8) = (0.13988739479107315472213342386758);
            wts(9) = (0.14452440398997005906382716655375);
            wts(10) = (0.14608113364969042719198514768337);
            wts(11) = (0.14452440398997005906382716655375);
            wts(12) = (0.13988739479107315472213342386758);
            wts(13) = (0.13226893863333746178105257449678);
            wts(14) = (0.12183141605372853419536717712573);
            wts(15) = (0.10879729916714837766347457807011);
            wts(16) = (0.093444423456033861553289741113932);
            wts(17) = (0.076100113628379302017051653300183);
            wts(18) = (0.057134425426857208283635826472448);
            wts(19) = (0.036953789770852493799950668299330);
            wts(20) = (0.016017228257774333324224616858471);
            break;
        case 42 :
        case 43 :
        	resize(wts, 22);
            wts(0) = (0.014627995298272200684991098047185);
            wts(1) = (0.033774901584814154793302246865913);
            wts(2) = (0.052293335152683285940312051273211);
            wts(3) = (0.069796468424520488094961418930218);
            wts(4) = (0.085941606217067727414443681372703);
            wts(5) = (0.10041414444288096493207883783054);
            wts(6) = (0.11293229608053921839340060742178);
            wts(7) = (0.12325237681051242428556098615481);
            wts(8) = (0.13117350478706237073296499253031);
            wts(9) = (0.13654149834601517135257383123152);
            wts(10) = (0.13925187285563199337541024834181);
            wts(11) = (0.13925187285563199337541024834181);
            wts(12) = (0.13654149834601517135257383123152);
            wts(13) = (0.13117350478706237073296499253031);
            wts(14) = (0.12325237681051242428556098615481);
            wts(15) = (0.11293229608053921839340060742178);
            wts(16) = (0.10041414444288096493207883783054);
            wts(17) = (0.085941606217067727414443681372703);
            wts(18) = (0.069796468424520488094961418930218);
            wts(19) = (0.052293335152683285940312051273211);
            wts(20) = (0.033774901584814154793302246865913);
            wts(21) = (0.014627995298272200684991098047185);
            break;
        case 44 :
        case 45 :
        	resize(wts, 23);
            wts(0) = (0.013411859487141772081309493458615);
            wts(1) = (0.030988005856979444310694219641885);
            wts(2) = (0.048037671731084668571641071632034);
            wts(3) = (0.064232421408525852127169615158911);
            wts(4) = (0.079281411776718954922892524742043);
            wts(5) = (0.092915766060035147477018617369765);
            wts(6) = (0.10489209146454141007408618501474);
            wts(7) = (0.11499664022241136494164351293396);
            wts(8) = (0.12304908430672953046757840067201);
            wts(9) = (0.12890572218808214997859533939979);
            wts(10) = (0.13246203940469661737164246470332);
            wts(11) = (0.13365457218610617535145711054584);
            wts(12) = (0.13246203940469661737164246470332);
            wts(13) = (0.12890572218808214997859533939979);
            wts(14) = (0.12304908430672953046757840067201);
            wts(15) = (0.11499664022241136494164351293396);
            wts(16) = (0.10489209146454141007408618501474);
            wts(17) = (0.092915766060035147477018617369765);
            wts(18) = (0.079281411776718954922892524742043);
            wts(19) = (0.064232421408525852127169615158911);
            wts(20) = (0.048037671731084668571641071632034);
            wts(21) = (0.030988005856979444310694219641885);
            wts(22) = (0.013411859487141772081309493458615);
            break;
        case 46 :
        case 47 :
        	resize(wts, 24);
            wts(0) = (0.012341229799987199546805667070037);
            wts(1) = (0.028531388628933663181307815951878);
            wts(2) = (0.044277438817419806168602748211338);
            wts(3) = (0.059298584915436780746367758500109);
            wts(4) = (0.073346481411080305734033615253117);
            wts(5) = (0.086190161531953275917185202983743);
            wts(6) = (0.097618652104113888269880664464247);
            wts(7) = (0.10744427011596563478257734244661);
            wts(8) = (0.11550566805372560135334448390678);
            wts(9) = (0.12167047292780339120446315347626);
            wts(10) = (0.12583745634682829612137538251118);
            wts(11) = (0.12793819534675215697405616522470);
            wts(12) = (0.12793819534675215697405616522470);
            wts(13) = (0.12583745634682829612137538251118);
            wts(14) = (0.12167047292780339120446315347626);
            wts(15) = (0.11550566805372560135334448390678);
            wts(16) = (0.10744427011596563478257734244661);
            wts(17) = (0.097618652104113888269880664464247);
            wts(18) = (0.086190161531953275917185202983743);
            wts(19) = (0.073346481411080305734033615253117);
            wts(20) = (0.059298584915436780746367758500109);
            wts(21) = (0.044277438817419806168602748211338);
            wts(22) = (0.028531388628933663181307815951878);
            wts(23) = (0.012341229799987199546805667070037);
            break;
        case 48 :
        case 49 :
        	resize(wts, 25);
            wts(0) = (0.011393798501026287947902964113235);
            wts(1) = (0.026354986615032137261901815295299);
            wts(2) = (0.040939156701306312655623487711646);
            wts(3) = (0.054904695975835191925936891540473);
            wts(4) = (0.068038333812356917207187185656708);
            wts(5) = (0.080140700335001018013234959669111);
            wts(6) = (0.091028261982963649811497220702892);
            wts(7) = (0.10053594906705064420220689039269);
            wts(8) = (0.10851962447426365311609395705012);
            wts(9) = (0.11485825914571164833932554586956);
            wts(10) = (0.11945576353578477222817812651290);
            wts(11) = (0.12224244299031004168895951894585);
            wts(12) = (0.12317605372671545120390287307905);
            wts(13) = (0.12224244299031004168895951894585);
            wts(14) = (0.11945576353578477222817812651290);
            wts(15) = (0.11485825914571164833932554586956);
            wts(16) = (0.10851962447426365311609395705012);
            wts(17) = (0.10053594906705064420220689039269);
            wts(18) = (0.091028261982963649811497220702892);
            wts(19) = (0.080140700335001018013234959669111);
            wts(20) = (0.068038333812356917207187185656708);
            wts(21) = (0.054904695975835191925936891540473);
            wts(22) = (0.040939156701306312655623487711646);
            wts(23) = (0.026354986615032137261901815295299);
            wts(24) = (0.011393798501026287947902964113235);
            break;
    }
}

KOKKOS_FUNCTION
void get_quadrature_gauss_legendre(
    const int order,
    int& nq,
    Kokkos::View<rtype**>::HostMirror& quad_pts,
    Kokkos::View<rtype*>::HostMirror& quad_wts) {

    Nodes::get_gauss_legendre_segment_nodes(
    	order, subview(quad_pts, ALL(), 0));
    get_segment_weights_gl(order, quad_wts);
    nq = quad_wts.extent(0);
}

} // end namespace Quadrature