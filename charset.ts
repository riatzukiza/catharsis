
export default class CharacterSet {
    private _charmap:object
    private _charset:string[]

    get charmap() {return this._charmap}
    get charset() {return this._charset}

    get size()    {return this.charset.length}

    constructor(text:string) {
        this._charmap = {}
        this._charset = []

        if(text)
            this.integrateText(text)
    }

    static create(text:string) {
        return new CharacterSet(text)
    }

    private integrateText(text:string) {
        for(let c of text) {
            if(this.charmap[c]) continue;
            this.charmap[c] = this.charset.length
            this.charset.push(c)
        }
    }

    getCharFromIndex(i:number):string {
        return this._charset[i]
    }
    getIndexFromChar(c:string):number {
        return this._charmap[c]
    }
}
