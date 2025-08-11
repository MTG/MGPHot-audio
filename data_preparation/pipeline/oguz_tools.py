import os
import sys
import re
from itertools import permutations

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

EXCLUDE_TITLES = {"full album"}

############################## Regex Functions for Youtube Metadata ##############################


ARTIST_JOIN_SYMBOLS = [",", "-", "&", "and", " "]
FEAT_START_SYMBOLS = ["featuring", "feat", "feat.", "ft.", "with", " "]

TITLE_JOIN_SYMBOLS = ["-", ":", " "]

# Create regex for joining names. Will match any single symbol, with 0 or 1 space around it
ARTIST_JOIN_PATTERN = r"\s?" + rf"({'|'.join(ARTIST_JOIN_SYMBOLS)})" + r"\s?"
FEAT_START_PATTERN = r"\s?" + rf"({'|'.join(FEAT_START_SYMBOLS)})" + r"\s?"
TITLE_JOIN_PATTERN = r"\s?" + rf"[{''.join(TITLE_JOIN_SYMBOLS)}]" + r"\s?"


def wrap_in_brackets(s):
    return r"[\[\(]?" + s + r"[\]\)]?" + r"\s?"


##################################### Cleaning Methods #####################################


def clean_uploader_name(v_uploader):
    """Cleans the uploader name (channel) from unnecessary information. Returns the cleaned
    uploader name."""

    # Uploader is a single entity not concatenated artists or channels
    v_uploader = re.sub(r"\s?-?\s?topic\Z", "", v_uploader)  # TODO: is there .topic?
    v_uploader = re.sub(r"vevo\Z", "", v_uploader)
    # v_uploader = re.sub(r"[^u][^n]official\Z", "", v_uploader) # TODO: there are unofficial channels as well
    v_uploader = re.sub(
        r"\s?[\[\(\{]?\s?(official|hq)\s?[\]\)\}]?", "", v_uploader
    )  # TODO???
    return v_uploader


def clean_video_title(v_title):
    """Cleans the video title from unnecessary information. Returns the cleaned
    video title."""

    v_title = re.sub(
        r"\s?[\[\(\{]\s?official\s?(hd)?\s?(audio|video|music)?\s?[\]\)\}]", "", v_title
    )  # TODO: official video?
    v_title = re.sub(
        r"\s?[\[\(\{]\s?official\s?(music|lyrics?|youtube)\s?video\s?[\]\)\}]",
        "",
        v_title,
    )
    v_title = re.sub(
        r"\s?[\[\(\{]\s?(explicit|uncensored|extended|clean)\s?(version)?[\]\)\}]",
        "",
        v_title,
    )
    v_title = re.sub(r"\s?[\[\(\{]\s?(lyrics?|video|audio)\s?[\]\)\}]", "", v_title)
    v_title = re.sub(
        r"\s?[\[\(\{]\s?music\s?(video|audio|lyrics?)?\s?[\]\)\}]", "", v_title
    )
    v_title = re.sub(r"\s?[\[\(\{]\s?lyrics?\s?(video|audio)?\s?[\]\)\}]", "", v_title)
    v_title = re.sub(r"\s?[\[\(\{]\s?visualizer\s?[\]\)\}]", "", v_title)
    v_title = re.sub(r"\s?[\[\(\{]\s?pseudo\s?video\s?[\]\)\}]", "", v_title)
    v_title = re.sub(r"\s?-\s?original(\s?version)?", "", v_title)

    v_title = re.sub(
        r"\s?[\[\(\{]\s?\d{0,4}\s?-?\s?remaster(ed)(\s?version)?\s?[\]\)\}]",
        "",
        v_title,
    )
    v_title = re.sub(
        r"\s?[\[\(\{]\s?remaster(ed)?(\s?version)?\s?-?\s?\d{0,4}\s?[\]\)\}]",
        "",
        v_title,
    )
    v_title = re.sub(
        r"\s?[\[\(\{]\s?remaster(ed)?\s?-?\s?\d{0,4}\s?(\s?version)?[\]\)\}]",
        "",
        v_title,
    )
    v_title = re.sub(
        r"\s?[\[\(\{]\s?remaster(ed)?\s?(in)?\s?(4k|8k)?[\]\)\}]", "", v_title
    )

    v_title = re.sub(r"\s?[\[\(\{]\s?(hq|hd)\s?[\]\)\}]", "", v_title)
    v_title = re.sub(
        r"\s?[\[\(\{]\s?(stereo|mono|original)(\s?version)?\s?[\]\)\}]", "", v_title
    )

    # Multiple spaces
    v_title = re.sub(r"\s{2,}", " ", v_title)
    # Trailing spaces
    v_title = re.sub(r"\s\Z", "", v_title)
    # Leading spaces
    v_title = re.sub(r"\A\s", "", v_title)
    return v_title


##################################### Youtube Search Algorithm #####################################


# TODO: Use feat_artist_names as normal artists too
def create_title_artist_combinations_regex(
    title: str, artist_names: list, feat_artist_names: list = []
):
    """
    Devuelve una lista de patrones regex compilados para matchear combinaciones comunes
    de título y artistas (con o sin artistas invitados), usados en títulos de vídeos de YouTube.
    """

    # Escape solo del contenido, no de los separadores como 'ft.', ':', '-'
    title = re.escape(title)
    artist_names = [re.escape(a) for a in artist_names]
    feat_artist_names = [re.escape(a) for a in feat_artist_names]

    # Todas las permutaciones de artistas principales
    artist_perms = list(permutations(artist_names))
    artist_patterns = [ARTIST_JOIN_PATTERN.join(a) for a in artist_perms]

    # Generar patrones de artistas invitados (con y sin paréntesis)
    feat_artist_patterns = []
    if len(feat_artist_names) == 1:
        raw_feat = FEAT_START_PATTERN + feat_artist_names[0]
        feat_artist_patterns = [
            wrap_in_brackets(raw_feat),  # (feat. X)
            raw_feat                     # feat. X
        ]
    elif len(feat_artist_names) > 1:
        feat_perms = list(permutations(feat_artist_names))
        for perm in feat_perms:
            raw_feat = FEAT_START_PATTERN + ARTIST_JOIN_PATTERN.join(perm)
            feat_artist_patterns.extend([
                wrap_in_brackets(raw_feat),
                raw_feat
            ])

    total_pattern = []

    if not feat_artist_patterns:
        # Sin featuring
        for artist_pattern in artist_patterns:
            total_pattern.extend([
                re.compile(artist_pattern + TITLE_JOIN_PATTERN + title, re.IGNORECASE),  # X - Title
                re.compile(title + TITLE_JOIN_PATTERN + artist_pattern, re.IGNORECASE)   # Title - X
            ])
    else:
        # Con featuring
        for feat_pat in feat_artist_patterns:
            total_pattern.append(re.compile(title + feat_pat, re.IGNORECASE))  # Title feat Y
            for artist_pattern in artist_patterns:
                total_pattern.extend([
                    re.compile(artist_pattern + feat_pat + TITLE_JOIN_PATTERN + title, re.IGNORECASE),   # X feat Y - Title
                    re.compile(title + TITLE_JOIN_PATTERN + artist_pattern + feat_pat, re.IGNORECASE),   # Title - X feat Y
                    re.compile(artist_pattern + TITLE_JOIN_PATTERN + title + feat_pat, re.IGNORECASE)    # X - Title feat Y
                ])

    return total_pattern


def prepare_track_for_query(track):
    """Adapta el diccionario del dataset con campos 'artist' y 'title'."""

    # Si el track tiene "track_artist_names", usamos el método original
    if "track_artist_names" in track and track["track_artist_names"] != []:
        key = "track_artist_names"
        t_artists = [a.lower() for a in track[key]]
        t_feat_artists = [a.lower() for a in track.get("track_feat_names", [])]
        t_title = track["track_title"].lower()
    else:
        # Modo compatible con el JSON de mgphot_gene_values.tsv
        t_artists = [track["artist"].lower()]
        t_feat_artists = []
        t_title = track["title"].lower()

    return t_title, t_artists, t_feat_artists


def create_query_string(track):
    """Creates a query string from the track information. Returns the query string."""

    t_title, t_artists, t_feat_artists = prepare_track_for_query(track)

    if t_feat_artists != []:
        return f"{', '.join(t_artists)} - {t_title} (featuring {', '.join(t_feat_artists)})"
    else:
        return f"{', '.join(t_artists)} - {t_title}"


def prepare_track_for_matching(track):

    t_title, t_artists, t_feat_artists = prepare_track_for_query(track)

    t_title = soft_clean_text(t_title)
    t_artists = [soft_clean_text(artist) for artist in t_artists]
    t_feat_artists = [soft_clean_text(artist) for artist in t_feat_artists]

    return t_title, t_artists, t_feat_artists


def check_officiality(uploader, description, track_artist=None):
    uploader = uploader.lower().strip()
    description = description.lower().strip()

    if re.search(r"\s?-?\s?topic\Z", uploader):
        return True
    elif uploader.endswith("vevo"):
        return True
    elif "provided to youtube by" in description:
        return True
    elif "auto-generated by youtube" in description:
        return True
    elif track_artist is not None and soft_clean_text(track_artist) in soft_clean_text(uploader):
        return True
    return False

import re
import unicodedata
import unidecode

##################################### Artist Relations #####################################


def collect_all_related_artists(total_artist_ids, artist_id, artists_dict):
    """Collects all the artist ids related to a given artist id. It includes
    the artist's aliases, the aliases of the artist's members, and the aliases
    of the artist's members. It also includes the members of the artist and
    the aliases of each member."""

    # Include the artist ID
    total_artist_ids.update({artist_id})

    # Some artists are not in the artist dictionary so we can
    # not get more information about them
    if artist_id in artists_dict:
        # Get the artist's information
        artist = artists_dict[artist_id]

        # Include the artist's aliases
        artist_aliases = artist.get("aliases", [])
        total_artist_ids.update(artist_aliases)

        # More information about the artist's aliases
        for alias_id in artist_aliases:
            # If an alias is a group, include its members
            alias_members = artists_dict[alias_id].get("members", [])
            total_artist_ids.update(alias_members)
            # and the aliases of each member
            for member_id in alias_members:
                total_artist_ids.update(artists_dict[member_id].get("aliases", []))

        # If the artist is a group add its members
        members = artist.get("members", [])
        total_artist_ids.update(members)

        # and the aliases of each member
        for member_id in members:
            total_artist_ids.update(artists_dict[member_id].get("aliases", []))
            # we make sure that members can not have members

        # If the artist has name variations, include them
        if "namevariations_id" in artist:
            namevar_ids = set(artist["namevariations_id"])
            total_artist_ids.update(namevar_ids)
            for namevar_id in namevar_ids:
                total_artist_ids.update(artists_dict[namevar_id].get("aliases", []))


def collect_performance_artists(track, artists_dict):
    """Collects all the relevant artist ids for a track or a list of tracks.
    If for a given track, artist_ids are not available, it uses the release
    artist_ids instead. It also includes the track featured artists if available.
    Using this set of artist_ids, it collects the IDs for the artists and
    their aliases. If an artist is a group, it also includes each member and
    their aliases."""

    # If a single track is given get its artists
    if type(track) is dict:
        # Determine which ID to use
        if track["track_artist_ids"] != []:
            ids = set(track["track_artist_ids"])
        else:
            ids = set(track["release_artist_ids"])
        # Include the featured artists
        ids.update(set(track["track_feat_ids"]))

        # Collect the IDs for all artists, their aliases and group members
        artist_ids = set()
        for id in ids:
            # Get the related artist ids
            collect_all_related_artists(artist_ids, id, artists_dict)
        return artist_ids

    # If a list of tracks is given, collect the artists for each track
    elif type(track) is list:
        artist_ids = set()
        for t in track:
            artist_ids.update(collect_performance_artists(t, artists_dict))
        return artist_ids
    else:
        raise TypeError(
            f"track must be a dict or a list of dicts, \
                        not {type(track)}"
        )


def collect_writer_artists(track, artists_dict):
    """Collects all the relevant writer artist ids for a track or a list of tracks.
    Using this set of artist_ids, it collects the IDs for the artists and
    their aliases. If an artist is a group, it also includes each member and
    their aliases."""

    # If a single track is given get its artists
    if type(track) is dict:
        ids = set(track["track_writer_ids"])

        # Collect the IDs for all artists, their aliases and group members
        artist_ids = set()
        for id in ids:
            # Get the related artist ids
            collect_all_related_artists(artist_ids, id, artists_dict)
        return artist_ids

    # If a list of tracks is given, collect the artists for each track
    elif type(track) is list:
        artist_ids = set()
        for t in track:
            artist_ids.update(collect_writer_artists(t, artists_dict))
        return artist_ids
    else:
        raise TypeError(
            f"track must be a dict or a list of dicts, \
                        not {type(track)}"
        )


##################################### Text Cleaning Methods #####################################


def is_latin_character(char):
    # Unicode ranges for Basic Latin and Latin-1 Supplement, Latin Extended-A, and more.
    # This covers the basic alphabet and extended characters with diacritics.
    latin_ranges = [
        (0x0041, 0x005A),  # Basic Latin uppercase A-Z
        (0x0061, 0x007A),  # Basic Latin lowercase a-z
        (0x00C0, 0x00D6),  # Latin-1 Supplement uppercase A-O with diacritics
        (
            0x00D8,
            0x00F6,
        ),  # Latin-1 Supplement uppercase O with diacritics and lowercase o-y
        (0x00F8, 0x00FF),  # Latin-1 Supplement lowercase o-y with diacritics
        (0x0100, 0x017F),  # Latin Extended-A
        (0x0180, 0x024F),  # Latin Extended-B
        # Additional ranges can be added for Latin Extended Additional, etc.
    ]

    code_point = ord(char)  # Get the Unicode code point of the character

    # Check if the character falls within any of the Latin ranges
    for start, end in latin_ranges:
        if start <= code_point <= end:
            return True

    # If the character does not fall within any range, it's not considered Latin
    return False


def remove_latin_diacritics(text):
    """Removes diacritics from Latin characters but do not alter other characters.
    Returns the text without diacritics."""

    result = []
    for char in text:
        # If the character is a Latin letter with a diacritic, remove the diacritic
        if unicodedata.category(char).startswith("L") and is_latin_character(char):
            char = unidecode.unidecode(char)
        result.append(char)
    return "".join(result)


def clean_parentheses(text):

    # Remove all parentheses and their content
    text = re.sub(r"\s\(.*?\)\Z", "", text)

    return text


def hard_clean_text(text):
    """Cleans the text from unnecessary information. Returns the cleaned text."""

    # Lowercase the text
    text = text.lower()

    # Remove the leading "the", "A", and "an" from the text
    text = re.sub(r"\A(the|a|an)\s", "", text)

    # Replace & with and
    text = re.sub(r"\s&\s", " and ", text)

    # Remove all punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    # Trailing spaces
    text = re.sub(r"\s\Z", "", text)
    # Leading spaces
    text = re.sub(r"\A\s", "", text)

    # Remove diacritics from Latin characters
    text = remove_latin_diacritics(text)

    return text


def soft_clean_text(text):
    """Cleans the text from unnecessary information. Returns the cleaned text."""

    # Lowercase the text
    text = text.lower()

    # Remove the leading "the", "A", and "an" from the text
    text = re.sub(r"\A(the|a|an)\s", "", text)

    # Replace & with and
    text = re.sub(r"\s&\s", " and ", text)

    result = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("L"):  # Check if it's a letter
            if is_latin_character(char):
                decoded = unidecode.unidecode(char)
                result.append(decoded)
            else:
                # We do not process the non-latin characters
                result.append(char)
        elif category.startswith("P"):  # Punctuation
            # Simplify the punctuation
            # Replace all dashes with a single dash
            char = re.sub(r"[―－‐‑‒–—﹘﹘﹣⁃]", "-", char)
            # Replace all quotes with a single quote
            char = re.sub(r'["‘’“”‚„‛‟]', "'", char)
            result.append(char)
        else:
            result.append(char)
        # else: # Remove
        #     continue # TODO ??
    return "".join(result)