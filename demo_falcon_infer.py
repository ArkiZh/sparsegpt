from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# model_path = "models/falcon-prune-0.5-2v4"
# model_path = "tiiuae/falcon-7b"
model_path = "models/falcon-prune-0.5"

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

def func():
    sequences = pipeline(
    """Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
    Daniel: Hello, Girafatron!
    Girafatron:""",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


for i in range(10):
    func()


ts = []
import time
for i in range(100):
    print("="*100)
    print(i)
    start = time.time()
    func()
    ts.append(time.time()-start)
    print(f"{i}: {ts[-1]}")
    
print(model_path, sum(ts)/len(ts), ts)



# models/falcon-prune-0.5-2v4 4.6892789101600645 [5.089198112487793, 5.085782051086426, 5.114762544631958, 5.109082937240601, 2.633437156677246, 5.1270976066589355, 5.067387819290161, 5.136662006378174, 5.075659275054932, 4.411386728286743, 5.129164934158325, 5.1074512004852295, 0.1677684783935547, 3.5530002117156982, 5.136714220046997, 5.144496202468872, 5.1726109981536865, 5.192942142486572, 5.167785882949829, 4.103496789932251, 5.100181818008423, 5.118628025054932, 5.130586862564087, 5.072350025177002, 5.068102598190308, 5.058507442474365, 5.074558734893799, 5.065417051315308, 5.090339183807373, 5.116801500320435, 5.076728343963623, 5.076451539993286, 5.09336256980896, 1.9362666606903076, 5.048927307128906, 5.058809518814087, 5.059404611587524, 4.968549013137817, 5.151345252990723, 5.17875862121582, 5.1581010818481445, 5.159265041351318, 5.1953699588775635, 5.187088489532471, 5.190167427062988, 5.176529407501221, 5.0851731300354, 5.114651203155518, 5.131534814834595, 4.649767637252808, 5.102085590362549, 5.136752367019653, 5.120728254318237, 1.8035101890563965, 5.089012384414673, 5.109593868255615, 5.053250312805176, 5.102139949798584, 5.114032745361328, 0.5056891441345215, 5.0919129848480225, 5.118748903274536, 5.095325708389282, 5.075154542922974, 5.081960916519165, 5.065951824188232, 5.082533121109009, 5.116877317428589, 5.074904441833496, 5.088106632232666, 5.129145860671997, 0.8636071681976318, 5.111386775970459, 4.840802907943726, 5.121263742446899, 4.061909437179565, 5.132555961608887, 5.151143789291382, 3.576157331466675, 5.095820426940918, 5.154868841171265, 5.135189771652222, 3.2618045806884766, 3.5925025939941406, 5.115600347518921, 0.9958608150482178, 5.087613105773926, 4.873794078826904, 5.1202216148376465, 5.043047666549683, 5.149886846542358, 5.078547954559326, 5.119497060775757, 4.978109121322632, 3.990652084350586, 1.3872427940368652, 5.030301809310913, 5.079145908355713, 5.139967918395996, 5.1643593311309814]
# tiiuae/falcon-7b 4.7146760892868045 [4.833833456039429, 5.145865201950073, 5.139381170272827, 5.135594367980957, 5.135425090789795, 5.131821870803833, 5.12780499458313, 5.1217427253723145, 1.381779432296753, 5.1215715408325195, 5.126841306686401, 5.120545864105225, 5.133934736251831, 5.156407356262207, 5.13863205909729, 5.1348512172698975, 5.113964557647705, 5.143542289733887, 5.108840465545654, 5.139296531677246, 5.128737449645996, 5.109914302825928, 5.047515869140625, 4.770932674407959, 4.7831501960754395, 4.806725263595581, 4.7818543910980225, 4.751766204833984, 3.444662094116211, 4.712087154388428, 4.676669597625732, 4.713046073913574, 4.693930387496948, 4.746452808380127, 4.775953054428101, 4.7319886684417725, 4.71325421333313, 4.708388566970825, 4.70278525352478, 4.715484619140625, 4.726037502288818, 4.7406322956085205, 4.738149166107178, 4.757812738418579, 4.7609381675720215, 4.783184766769409, 4.724225759506226, 3.88773512840271, 4.721215724945068, 4.756110906600952, 4.743828058242798, 4.755292892456055, 4.760598421096802, 4.748075008392334, 4.747446060180664, 4.743723154067993, 4.7196784019470215, 4.752886533737183, 4.773076057434082, 4.757158041000366, 4.769305467605591, 4.831469535827637, 4.746230125427246, 4.979582786560059, 4.810016870498657, 4.798813819885254, 3.3626179695129395, 4.790170907974243, 4.782728910446167, 4.799213886260986, 4.749988794326782, 4.77739405632019, 4.7600791454315186, 4.7777931690216064, 4.804382801055908, 4.7862608432769775, 4.775851726531982, 4.74959397315979, 4.723509311676025, 4.733578443527222, 4.75012731552124, 4.725043535232544, 4.65705132484436, 0.39917421340942383, 4.672913074493408, 4.664682865142822, 4.646974563598633, 4.6503190994262695, 4.821235179901123, 4.772396564483643, 4.684001207351685, 4.705363512039185, 4.6853697299957275, 4.733177900314331, 4.722799062728882, 4.734105825424194, 4.787352800369263, 4.734036207199097, 4.709203243255615, 4.698919296264648]
# models/falcon-prune-0.5 4.765863358974457 [4.890791177749634, 4.895196914672852, 4.927488327026367, 4.883697032928467, 4.938307762145996, 4.8772053718566895, 4.886040210723877, 4.8343188762664795, 4.879843711853027, 4.858964920043945, 4.876830339431763, 4.865772485733032, 4.850574731826782, 4.874675035476685, 4.894918918609619, 4.854320049285889, 4.892954587936401, 2.135056495666504, 1.9860942363739014, 4.8551812171936035, 4.889811754226685, 4.8837058544158936, 4.899036884307861, 4.867233037948608, 4.90283203125, 4.9102747440338135, 4.93451714515686, 4.913348197937012, 4.878737926483154, 4.889258623123169, 4.937045574188232, 4.907260417938232, 4.886576890945435, 4.879697799682617, 4.88376784324646, 4.884681701660156, 2.3935070037841797, 4.870271682739258, 4.858092308044434, 4.881659269332886, 4.894749641418457, 4.88767409324646, 4.854605197906494, 4.865352630615234, 4.891340255737305, 4.8895227909088135, 4.871072053909302, 4.906238079071045, 4.896203994750977, 4.895479202270508, 4.863914966583252, 4.877811908721924, 4.874040365219116, 4.839585065841675, 4.85626220703125, 4.872990131378174, 4.8667075634002686, 4.844642162322998, 4.88377046585083, 4.872897148132324, 4.844428777694702, 4.842294931411743, 4.865355968475342, 4.866680383682251, 4.888078927993774, 4.886795282363892, 4.870201349258423, 4.892158269882202, 4.879223585128784, 4.878084659576416, 4.873475790023804, 4.877955198287964, 4.872257232666016, 4.828287839889526, 4.884273052215576, 4.858088970184326, 4.879279613494873, 4.91664719581604, 4.8957459926605225, 4.855912446975708, 4.857800245285034, 4.8933751583099365, 4.913161277770996, 4.893325090408325, 4.91671085357666, 4.902323246002197, 4.908832788467407, 4.917587757110596, 4.856094598770142, 4.889703035354614, 4.831254243850708, 4.904514789581299, 4.849622964859009, 4.821035385131836, 1.6694657802581787, 4.850921154022217, 4.88009786605835, 4.874447584152222, 4.855086326599121, 4.83134126663208] 


# origin
# Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron: Hello, Daniel.
# Daniel: How's it going?
# Girafatron: It's going fine, Daniel, thanks. So how's it going?
# Daniel: Fine!
# Girafatron: Oh.
# Daniel: What?!
# Girafatron: (sighs) Nothing!
# Daniel: …Are you okay?
# Girafatron: …Yes.
# Daniel: (nervously) Good.
# Girafatron: I was just asking, you know.
# Daniel: Yeah, that's fine. I'm asking too.
# Girafatron:

# origin
# Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron: Hello, Daniel. Welcome to the Girafatron website!
# Daniel: You have to be the craziest person I've ever interviewed! You are all over the place!
# Girafatron: Thank you very much, Daniel!
# Daniel: You're a man of few, but very strong words.
# Girafatron: Yes, I do have a very small amount of words.
# Daniel: I have no doubt of that. But let me ask you a serious question. You have a website devoted to a single subject...
# Girafatron: The most glorious animal known to man: The Giraffe!
# Daniel: Yes

# prune 0.5
# Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron: How may I help you today?
# Daniel: You are a Girafatron, aren't you?
# Girafatron: I am a giraffaton! What is your name?
# Daniel: I am a name! What am I called?
# Girafatron: You are a name! You are a name!
# Daniel: I am a name. What are you doing?
# Girafatron: I am doing a name! I have a name.
# Daniel: I am a name! I am a name!
# Girafotron: You have a name! You have a name!
# Daniel: I am a

# prune 0.5
# Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron: Hello Daniel! What can I do for you?
# Daniel: I want to know what you have for me.
# Girafatron: I have a giraffe!
# Daniel: You have a giraffe? I want one!
# Girafatron: Well Daniel, you’ll need to wait for your next trip to Africa.
# Daniel: I’m going to Africa? I’ll wait for the giraffe then. How much are giraffes?
# Girafatron: $5
# Daniel: Okay!
# Girafatron: Giraffe is $15!
# Daniel: Giraffes are $15?!

# prune 0.5 2v4
# Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron: Hello, Daniel. Daniel is the name of a girafastron that is very happy, but he is not yet a girafastron because he has a girafafarist. Daniel is a very good name, and I am very happy with Daniel.
# Daniel: Yes, Giraftron. I agree with Daniel that girafafarists are the finest name I ever heard because it has been my favorite name ever.
# Girafatron: I agree with Daniel because we have a very good name that is a great name because Daniel has been the finest name ever.
# Daniel: I agree with Giraftron because he is the girafaf

# prune 0.5 2v4
# Result: Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron: Hello! Girafatron! Girafatron! Girafatron!
# Daniel: I have been watching your videos for quite some time. I am glad you are a part of my video channel. Thank you for your videos!
# Girafatron: I am delighted you are viewing my videos!
# Daniel: I am glad you are part of my video channel. Thank you for your videos!
# Girafatron: I am delighted you are viewing my videos!

