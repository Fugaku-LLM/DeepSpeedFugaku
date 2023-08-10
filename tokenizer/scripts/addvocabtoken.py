import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model
import argparse

# https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto#L289
spm_typedef = {
    'NORMAL' : 1,      
    'UNKNOWN' : 2,     
    'CONTROL' : 3,     
    'USER_DEFINED' : 4,
    'BYTE' : 6,        
    'UNUSED' : 5, 
}

def add(merry,piece,score,ntype):
    onepiece = spm.sentencepiece_model_pb2.ModelProto().SentencePiece(
        piece=piece,
        score=score,
        type=ntype)
    merry.pieces.append(onepiece)
    return merry

def main():
    parser = argparse.ArgumentParser(description="Process model files and other options.")
    
    parser.add_argument("model", help="Path to the .model file")
    parser.add_argument("out", help="Path to the output .model file")

    parser.add_argument("--piece", type=str, help="Additional piece information")
    parser.add_argument("--type", type=str, default="USER_DEFINED", help="Additional type information")
    parser.add_argument("--score", type=float, help="SentencePiece score (default for control token scores)", required=False)

    args = parser.parse_args()

    merry = model.ModelProto()
    merry.ParseFromString(open(args.model, 'rb').read())

    if args.score is None:
        # print(type(args.score))
        assert merry.pieces[1].piece=='<s>'
        score = merry.pieces[1].score
    else:
        score = float(args.score)
    ntype = spm_typedef[args.type]

    add(merry,args.piece,score,ntype)

    with open(args.out, 'wb') as f:
        f.write(merry.SerializeToString())

if __name__ == "__main__":
    main()