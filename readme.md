# Catharsis

# What is it?

Catharsis is the purification and purgation of emotions—particularly pity and fear—through art or any extreme change in emotion that results in renewal and restoration.


## Ideas

So it should be possible to create a system that scrapes twitter streams, 
extracting relavent terminology from an initial seed term (A place to start searching, maybe the fire hose?).

The firehose is an enterprise grade service that twitter offers, I will likely not be able to get access to this in the beginning.

We can decide where to search next after some amount of time has passed (Cron job?) from a probability distrobution, select a new search term(s)

## Personality

Personality will be based on the big 5, and the model will contain an internal state which indicates where on a spectrum it falls on each of the 5 personality traits.

In addition to 5 values indicating an overall personality moving in one way or another, there will be 5 values which can flex around based on events.

We will have models trained for each of the 5 traits, and based on a combination of 

### Extraversion
is the state of primarily obtaining gratification outside of onesself 
### Agreeableness

### Openness

### Neuroticisim

### Conscientiousness

## Oppinion

An oppinion is a perspective that an individual holds regardless of knowledge or facts.

## Persona

A persona is the outward expression of an individuals personality.

## Accounts

An account represents credentials to access a social media platform assuming a specific identity.

## Language model

A language model is a basis for creating words, and navigating thoughts, in a way that is transmissable. It allows personas to share state 
with each another.


# Some important commands 

```
docker build --tag tfjs-gpu .
docker run -it --rm tfjs-gpu --volume $CWD:/app
docker run -it --gpus=all --rm --volume ${PWD}:/app tfjs-gpu


```
